# -*- encoding: utf-8 -*-
# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PointWiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs



class Bert4Rec(nn.Module):

    def __init__(self, user_num, item_num, device, args):
        
        super(Bert4Rec, self).__init__()

        self.user_num = user_num
        self.item_num = item_num
        self.dev = device
        self.mask_token = item_num + 1
        self.num_heads = args.num_heads

        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num+2, args.hidden_size, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_len+100, args.hidden_size) # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList() # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)

        for _ in range(args.trm_num):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer =  torch.nn.MultiheadAttention(args.hidden_size,
                                                            args.num_heads,
                                                            args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_size, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_size, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()
        self.projector = nn.Linear(args.hidden_size, args.hidden_size)
        self.loss_func = torch.nn.BCEWithLogitsLoss()


    def log2feats(self, log_seqs, positions):

        seqs = self.item_emb(log_seqs)
        #seqs = F.normalize(seqs, p=2, dim=-1)
        seqs *= self.item_emb.embedding_dim ** 0.5  # QKV/sqrt(D)
        #positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        #seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs += self.pos_emb(positions.long())
        seqs = self.emb_dropout(seqs)

        #timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        timeline_mask = (log_seqs == 0)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim

        #tl = seqs.shape[1] # time dim len for enforce causality
        #attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        #mask = (log_seqs < 1).unsqueeze(1).repeat(1, log_seqs.size(1), 1)
        #mask = mask.repeat(self.num_heads, 1, 1)

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,)
                                            #attn_mask=mask)
                                            #key_padding_mask=timeline_mask)
                                            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs) # (U, T, C) -> (U, -1, C)

        return log_feats


    def forward(self, log_seqs, pos_seqs, neg_seqs, positions): # for training        

        log_feats = self.log2feats(log_seqs, positions) # (bs, max_len, hidden_size)
        mask_index = torch.where(pos_seqs>0)
        log_feats = log_feats[mask_index] # (bs, mask_num, hidden_size)
        #log_feats = self.projector(log_feats)

        pos_embs = self.item_emb(pos_seqs) # (bs, mask_num, hidden_size)
        neg_embs = self.item_emb(neg_seqs) # (bs, mask_num, hidden_size)
        pos_embs = pos_embs[mask_index]
        neg_embs = neg_embs[mask_index]

        #log_feats = F.normalize(log_feats, p=2, dim=1)
        #pos_embs = F.normalize(pos_embs, p=2, dim=1)
        #neg_embs = F.normalize(neg_embs, p=2, dim=1)

        pos_logits = torch.mul(log_feats, pos_embs).sum(dim=-1) # (bs, mask_num)
        neg_logits = torch.mul(log_feats, neg_embs).sum(dim=-1) # (bs, mask_num)

        # pos_pred = self.pos_sigmoid(pos_logits)
        # neg_pred = self.neg_sigmoid(neg_logits)
        pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.dev), torch.zeros(neg_logits.shape, device=self.dev)
        pos_loss, neg_loss = self.loss_func(pos_logits, pos_labels), self.loss_func(neg_logits, neg_labels)
        loss = pos_loss + neg_loss

        return loss # loss


    def predict(self, log_seqs, item_indices, positions): # for inference

        log_seqs = torch.cat([log_seqs, self.mask_token * torch.ones(log_seqs.shape[0], 1, device=self.dev)], dim=1)
        pred_position = positions[:, -1] + 1
        positions = torch.cat([positions, pred_position.unsqueeze(1)], dim=1)
        log_feats = self.log2feats(log_seqs[:, 1:].long(), positions[:, 1:].long()) # user_ids hasn't been used yet
        #log_feats = self.log2feats(log_seqs[:, 1:].long(), positions)
        #log_feats = self.projector(log_feats)

        final_feat = log_feats[:, -1, :] # only use last QKV classifier, a waste

        item_embs = self.item_emb(item_indices) # (U, I, C)

        #log_feats = F.normalize(log_feats, p=2, dim=1)
        #item_embs = F.normalize(item_embs, p=2, dim=1)

        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)

        # preds = self.pos_sigmoid(logits) # rank same item list for different users

        return logits # preds # (U, I)



