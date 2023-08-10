# -*- encoding: utf-8 -*-
# here put the import lib
import copy
import random
import numpy as np
from torch.utils.data import Dataset
from utils.utils import random_neq


class SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation'''

    def __init__(self, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        pos = inter[-1]
        neg = []
        for _ in range(self.neg_num):
            per_neg = random_neq(1, self.item_num+1, non_neg)
            neg.append(per_neg)
            non_neg.append(per_neg)
        neg = np.array(neg)
        #neg = random_neq(1, self.item_num+1, inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions



class Seq2SeqDataset(Dataset):
    '''The train dataset for Sequential recommendation with seq-to-seq loss'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.aug_seq = args.aug_seq
        self.aug_seq_len = args.aug_seq_len


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break

        if self.aug_seq:
            seq_len = len(inter)
            pos[:- (seq_len - self.aug_seq_len) + 1] = 0
            neg[:- (seq_len - self.aug_seq_len) + 1] = 0
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        return seq, pos, neg, positions



class BertTrainDataset(Dataset):
    '''The train dataset for Bert4Rec'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.mask_token = item_num + 1


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        seq = self.data[index]

        tokens = []
        labels, neg_labels = [], []
        for s in seq:
            prob = random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(random.randint(1, self.item_num))
                else:
                    tokens.append(s)

                labels.append(s)
                neg = random_neq(1, self.item_num+1, seq)
                neg_labels.append(neg)

            else:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos

        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos)



class BertRecTrainDataset(Dataset):
    '''The train dataset for Bert4Rec'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_prob = args.mask_prob
        self.mask_token = item_num + 1


    def __len__(self):

        return 2 * len(self.data)

    def __getitem__(self, index):

        tokens = []
        labels, neg_labels = [], []

        if index >= len(self.data):
            seq = self.data[index - len(self.data)]
            for s in seq:
                tokens.append(s)
                labels.append(0)
                neg_labels.append(0)
            labels[-1] = tokens[-1]
            neg_labels[-1] = random_neq(1, self.item_num+1, seq)
            tokens[-1] = self.mask_token

        else:
            seq = self.data[index]
   
            for s in seq:
                prob = random.random()
                if prob < self.mask_prob:
                    prob /= self.mask_prob

                    if prob < 0.8:
                        tokens.append(self.mask_token)
                    elif prob < 0.9:
                        tokens.append(random.randint(1, self.item_num))
                    else:
                        tokens.append(s)

                    labels.append(s)
                    neg = random_neq(1, self.item_num+1, seq)
                    neg_labels.append(neg)

                else:
                    tokens.append(s)
                    labels.append(0)
                    neg_labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos

        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos)
        

class BertFinetuneDataset(Dataset):
    '''The train dataset for Bert4Rec'''

    def __init__(self, args, data, item_num, max_len, neg_num=1):
        
        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_token = item_num + 1


    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        tokens = []
        labels, neg_labels = [], []

        seq = self.data[index]
        for s in seq:
            tokens.append(s)
            labels.append(0)
            neg_labels.append(0)
        labels[-1] = tokens[-1]
        neg_labels[-1] = random_neq(1, self.item_num+1, seq)
        tokens[-1] = self.mask_token

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]
        neg_labels = neg_labels[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        neg_labels = [0] * mask_len + neg_labels
        pos = [0] * mask_len + pos

        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos)


class CL4SRecDataset(Dataset):

    def __init__(self, args, data, item_num, max_len, neg_num=1) -> None:
        '''
        The augment part refers to: https://github.com/RuihongQiu/DuoRec/blob/master/recbole/model/sequential_recommender/cl4srec.py
        '''
        super().__init__()

        super().__init__()
        self.data = data
        self.item_num = item_num
        self.max_len = max_len
        self.neg_num = neg_num
        self.mask_token = item_num + 1
        self.mask_crop_ratio = args.mask_crop_ratio

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        tokens = []

        # The original sequence
        seq = self.data[index]
        for s in seq:
            tokens.append(s)
        labels = tokens[-1]
        neg_labels = random_neq(1, self.item_num+1, seq)
        tokens[-1] = self.mask_token

        tokens = tokens[-self.max_len:]
        pos = list(range(1, len(tokens)+1))
        pos= pos[-self.max_len:]

        mask_len = self.max_len - len(tokens)
        
        tokens = [0] * mask_len + tokens
        pos = [0] * mask_len + pos

        aug_flag1 = random.random()
        if aug_flag1 <=0.33:
            aug1 = self.get_mask_seqs(seq)
        elif aug_flag1 <= 0.66:
            aug1 = self.get_crop_seqs(seq)
        else:
            aug1 = self.get_reorder_seqs(seq)

        aug_flag2 = random.random()
        if aug_flag2 <=0.33:
            aug2 = self.get_mask_seqs(seq)
        elif aug_flag2 <= 0.66:
            aug2 = self.get_crop_seqs(seq)
        else:
            aug2 = self.get_reorder_seqs(seq)


        return np.array(tokens), np.array(labels), np.array(neg_labels), np.array(pos), np.array(aug1), np.array(aug2)


    def get_mask_seqs(self, seq):

        mask_tokens = []

        for s in seq:
            prob = random.random()
            if prob < self.mask_crop_ratio:
                prob /= self.mask_crop_ratio

                if prob < 0.8:
                    mask_tokens.append(self.mask_token)
                elif prob < 0.9:
                    mask_tokens.append(random.randint(1, self.item_num))
                else:
                    mask_tokens.append(s)

            else:
                mask_tokens.append(s)
        
        mask_tokens = mask_tokens[-self.max_len:]
        mask_len = self.max_len - len(mask_tokens)
        mask_tokens = [0] * mask_len + mask_tokens

        return mask_tokens
    

    def get_crop_seqs(self, seq):

        crop_tokens = []

        crop_len = int(len(seq) * self.mask_crop_ratio)
        start_index = random.randint(0, len(seq) - crop_len - 1)

        for i in range(start_index, start_index + crop_len):

            crop_tokens.append(seq[i])
        
        crop_tokens = crop_tokens[-self.max_len:]
        mask_len = self.max_len - len(crop_tokens)
        crop_tokens = [0] * mask_len + crop_tokens

        return crop_tokens
    

    def get_reorder_seqs(self, seq):
        '''reorder subsequence in the original sequence'''
        num_reorder = int(len(seq) * self.mask_crop_ratio)
        reorder_begin = random.randint(0, len(seq) - num_reorder)
        shuffle_index = list(range(reorder_begin, reorder_begin + num_reorder))
        random.shuffle(shuffle_index)
        new_seq = copy.deepcopy(seq)
        new_seq[reorder_begin:reorder_begin + num_reorder] = np.array(seq)[shuffle_index]

        reorder_tokens = []

        for i in range(len(new_seq)):

            reorder_tokens.append(new_seq[i])

        reorder_tokens = reorder_tokens[-self.max_len:]
        mask_len = self.max_len - len(reorder_tokens)
        reorder_tokens = [0] * mask_len + reorder_tokens

        return reorder_tokens



class Seq2SeqCL4SRecDataset(CL4SRecDataset):

    def __init__(self, args, data, item_num, max_len, neg_num=1) -> None:

        super().__init__(args, data, item_num, max_len, neg_num)


    def __getitem__(self, index):

        inter = self.data[index]
        non_neg = copy.deepcopy(inter)
        
        seq = np.zeros([self.max_len], dtype=np.int32)
        pos = np.zeros([self.max_len], dtype=np.int32)
        neg = np.zeros([self.max_len], dtype=np.int32)
        nxt = inter[-1]
        idx = self.max_len - 1
        for i in reversed(inter[:-1]):
            seq[idx] = i
            pos[idx] = nxt
            neg[idx] = random_neq(1, self.item_num+1, non_neg)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        if len(inter) > self.max_len:
            mask_len = 0
            positions = list(range(1, self.max_len+1))
        else:
            mask_len = self.max_len - (len(inter) - 1)
            positions = list(range(1, len(inter)-1+1))
        
        positions= positions[-self.max_len:]
        positions = [0] * mask_len + positions
        positions = np.array(positions)

        aug_flag1 = random.random()
        if aug_flag1 <=0.33:
            aug1 = self.get_mask_seqs(inter)
        elif aug_flag1 <= 0.66:
            aug1 = self.get_crop_seqs(inter)
        else:
            aug1 = self.get_reorder_seqs(inter)

        aug_flag2 = random.random()
        if aug_flag2 <=0.33:
            aug2 = self.get_mask_seqs(inter)
        elif aug_flag2 <= 0.66:
            aug2 = self.get_crop_seqs(inter)
        else:
            aug2 = self.get_reorder_seqs(inter)


        return np.array(seq), np.array(pos), np.array(neg), np.array(positions), np.array(aug1), np.array(aug2)