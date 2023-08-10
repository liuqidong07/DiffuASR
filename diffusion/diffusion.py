# -*- encoding: utf-8 -*-
# here put the import lib
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from diffusion.utils import q_xt_x0, p_xt, normal_kl, mean_flat
from diffusion.transformer import TrmModel
from diffusion.gru import GRUModel, GuidedGRUModel
from diffusion.unet import UNet
from diffusion.unet1d import UNet1d
from diffusion.unet2d import UNet2d
from diffusion.wavenet import DiffWave
from diffusion.classifier import BertClassifier
from diffusion.preference import BertPref


class DiffusionModel(nn.Module):

    def __init__(self, user_num, item_num, device, args) -> None:
        
        super().__init__()
        self.set_config()
        self.config.model.in_channels = args.aug_num
        self.config.model.out_ch = args.aug_num
        self.config.num_diffusion_timesteps = args.num_diffusion_timesteps
        self.config.hidden_size = args.hidden_size
        self.n_steps = args.num_diffusion_timesteps
        self.batch_size = args.train_batch_size
        self.seq_len = args.aug_num
        self.hidden_size = args.hidden_size
        self.dm_scheduler = args.dm_scheduler
        self.device = device
        self.get_alpha()
        self.eta = 0.0
        self.timesteps = 5
        self.noise_model = args.noise_model
        self.rec_path = args.rec_path
        self.mask_token = item_num + 1
        self.rounding = args.rounding
        self.simple = args.simple

        if self.noise_model == "gru":
            self.unet = GuidedGRUModel(args)
        elif self.noise_model == "trm":
            self.unet = TrmModel(user_num, item_num, device, args)
        elif self.noise_model == "unet":
            self.unet = UNet(self.config, args.guide)
        elif self.noise_model == "unet1d":
            self.unet = UNet1d(self.config, args.guide)
        elif self.noise_model == "unet2d":
            self.unet = UNet2d(self.config, args.guide)
        elif self.noise_model == 'wave':
            self.unet = DiffWave(self.config)
        else:
            NotImplementedError
        
        self.item_emb = torch.nn.Embedding(item_num+2, args.hidden_size)
        
        self.pref = args.pref
        if self.pref:
            pref_model = BertPref(user_num, item_num, device, args)
            self.pref_model = self.load_preference(pref_model)

    

    def forward(self, seqs, guidance):

        t= torch.randint(low=0, high=self.n_steps, size=(self.batch_size // 2 + 1,)).to(self.device) # bs/2+1 random select t for half batch of data
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:self.batch_size] # (bs), get the random steps for the whole batch

        x0 = self.item_emb(seqs)    # (bs, aug_len, hidden_size)
        x0 = x0.unsqueeze(1)    # (bs, 1, aug_len, hidden_size)

        if self.pref:
            mask = self.mask_token * torch.ones(guidance.shape[0], 1, device=self.device)
            mask_emb = self.item_emb(mask.long())
            guide_vector = self.pref_model(guidance, self.item_emb(guidance), mask_emb)
        else:
            guidance = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)   # (bs, max_seq_len, hidden_size) 
            guide_vector = torch.sum(guidance, dim=1)   # (bs, hidden_size)

        xt, noise = q_xt_x0(x0, t, self.alpha_bar)  # xt: (bs, 1, seq_len, hidden_size), noise: (bs, 1, seq_len, hidden_size)
        #pred_noise = self.unet(xt.squeeze().float(), t, guide_vector)
        pred_noise = self.unet(xt.squeeze().float(), t, guide_vector)

        loss = F.mse_loss(noise.float(), pred_noise.unsqueeze(1))

        return loss

    
    def predict(self, guidance, item_indices):

        skip = self.n_steps // self.timesteps
        seq = range(0, self.n_steps, skip)

        x = torch.randn(guidance.shape[0], 1, self.seq_len, self.hidden_size).to(self.device)  # (bs, 1, seq_len, hidden_size)

        # get the guide vector
        if self.pref:
            mask = self.mask_token * torch.ones(guidance.shape[0], 1, device=self.device)
            mask_emb = self.item_emb(mask.long())
            guide_vector = self.pref_model(guidance, self.item_emb(guidance), mask_emb)
        else:
            guidance = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)   # (bs, max_seq_len, hidden_size) 
            guide_vector = torch.sum(guidance, dim=1)   # (bs, hidden_size)

        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            with torch.no_grad():
                #pred_noise = self.unet(x.squeeze(), t, guide_vector)
                pred_noise = self.unet(x.squeeze(), t, guide_vector)
                x = p_xt(x, pred_noise.unsqueeze(1), t, next_t, self.beta, self.eta)    # x: (bs, 1, seq_len, hidden_size)
        
        item_embs = self.item_emb(item_indices)
        x = x.squeeze()

        logits = []
        for i in range(self.seq_len):
            temp = x[:, i, :].unsqueeze(1).repeat(1, item_embs.shape[1], 1)
            logit = F.cosine_similarity(temp, item_embs, dim=-1)  # (bs, item_num)
            logits.append(logit)

        return logits

    
    def get_alpha(self):

        scale = 1000 / self.n_steps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02

        if self.dm_scheduler == 'linear':
            self.beta = torch.linspace(beta_start, beta_end, self.n_steps) # 等间隔的
        elif self.dm_scheduler == "quad":
            self.beta = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, self.n_steps) ** 2 # 每个时间步平方
        elif self.dm_scheduler == "sigmoid":
            self.beta = torch.linspace(-6, 6, self.n_steps)
            self.beta = torch.sigmoid(self.beta) * (beta_end - beta_start) + beta_start    # 把时间步归到-1~1之间
        elif self.dm_scheduler == "cosine":
            steps = self.n_steps + 1
            x = torch.linspace(0, self.n_steps, steps)
            alphas_cumprod = torch.cos(((x / self.n_steps) + 0.008) / (1 + 0.008) * torch.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            self.beta = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(self.beta, 0.0001, 0.9999)

        #self.beta = torch.linspace(beta_start, beta_end, self.n_steps).to(self.device)
        self.beta = self.beta.to(self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    
    def token_discrete_loss(self, x_t, input_ids):
        
        logits = self.lm_head(x_t)  # bsz, seqlen, vocab
        # print(logits.shape)
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        decoder_nll = loss_fct(logits.view(-1, logits.size(-1)), input_ids.view(-1)).view(input_ids.shape)
        # print(decoder_nll.shape)
        decoder_nll = decoder_nll.mean(dim=-1)
        decoder_nll = torch.mean(decoder_nll)   # add the mean by myself. I don't understand why not mean
        return decoder_nll
    

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.

        This term can't be optimized, as it only depends on the encoder.

        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)


    def get_x_start(self, x_start_mean, std):
        '''
        Using the interpolating policy OR using the convolution policy...
        :param x_start_mean:
        :return:
        '''
        noise = torch.randn_like(x_start_mean)
        # print(std.shape, noise.shape, x_start_mean.shape)
        assert noise.shape == x_start_mean.shape
        # print(x_start_mean.device, noise.device)
        return (
             x_start_mean + std * noise
        )


    def set_config(self):
        
        args={
            'data':
                {
                # 'dataset': '',
                'image_size': 8,
                # 'channels': 3,
                # 'logit_transform': False,
                # 'uniform_dequantization': False,
                # 'gaussian_dequantization': False,
                # 'random_flip': True,
                # 'rescaled': True,
                # 'num_workers': True,
                },

            'model':
                {'type': "simple",
                'in_channels': 10, # unet & unet1d: (num_channel==aug_seq_len)
                'out_ch': 10,  # unet & unet1d: (num_channel==aug_seq_len)
                'in_channels_2d': 1, # unet2d
                'out_ch_2d': 1,  # unet2d
                'ch': 8,
                'ch_mult': [1, 2, 2, 2],
                'num_res_blocks': 2,
                'attn_resolutions': [16, ],
                'dropout': 0.1,
                'var_type': 'fixedlarge',
                'ema_rate': 0.9999,
                'ema': True,
                'resamp_with_conv': True,},

            'diffusion':
                {
                # 'beta_schedule': 'linear',
                # 'beta_start': 0.0001,
                # 'beta_end': 0.02,
                'num_diffusion_timesteps': 1000,
                },

            'training':
                {
                # 'batch_size': 128,
                # 'n_epochs': 100,
                # 'n_iters': 5000000,
                # 'snapshot_freq': 5000,
                # 'validation_freq': 2000,
                },

            'sampling':
                {
                # 'batch_size': 64,
                # 'last_only': True,
                },

            'optim':
                {
                # 'weight_decay': 0.000,
                # 'optimizer': "Adam",
                # 'lr': 0.0002,
                # 'beta1': 0.9,
                # 'amsgrad': False,
                # 'eps': 0.00000001,
                # 'grad_clip': 1.0,
                }
            }

        temp={}
        for k,v in args.items():
            temp[k]=SimpleNamespace(**v)

        self.config = SimpleNamespace(**temp)


    def load_preference(self, model):

        checkpoint_path = os.path.join(self.rec_path, 'pytorch_model.bin')

        model_dict = model.state_dict()
        # To be compatible with the new and old version of model saver
        try:
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        except:
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device)

        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)

        model.load_state_dict(model_dict)

        return model





class DiffusionModel_CG(DiffusionModel):

    def __init__(self, user_num, item_num, device, args) -> None:

        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.classifier_scale = args.classifier_scale
        self.rec_path = args.rec_path
        self.dev = device
        self.guide_type = args.guide_type
        classifier = BertClassifier(user_num, item_num, device, args)
        self.classifier = self.load_classifier(classifier)

    
    def load_classifier(self, model):

        checkpoint_path = os.path.join(self.rec_path, 'pytorch_model.bin')

        model_dict = model.state_dict()
        # To be compatible with the new and old version of model saver
        try:
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        except:
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device)

        new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        model_dict.update(new_dict)

        model.load_state_dict(model_dict)

        return model
    

    def cond_fn(self, x, y=None):

        assert y is not None
        with torch.enable_grad():
            x_in = x.squeeze().detach().requires_grad_(True)
            #pos_emb = self.item_emb(y).detach().requires_grad_(True)
            mask = self.mask_token * torch.ones(x_in.shape[0], 1, device=self.dev)
            mask_emb = self.item_emb(mask.long()).detach().requires_grad_(True)
            #loss = self.classifier(x_in, pos_emb, mask_emb)
            loss = self.classifier(x_in, y, mask_emb)
            
            return torch.autograd.grad(loss.sum(), x_in)[0] * self.classifier_scale
    

    def forward(self, seqs, guidance):

        t= torch.randint(low=0, high=self.n_steps, size=(self.batch_size // 2 + 1,)).to(self.device) # bs/2+1 random select t for half batch of data
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:self.batch_size] # (bs), get the random steps for the whole batch

        x0 = self.item_emb(seqs)    # (bs, aug_len, hidden_size)
        x0 = x0.unsqueeze(1)    # (bs, 1, aug_len, hidden_size)

        # get the classifier label
        #first_index = -torch.sum(guidance>0, dim=-1)
        #y = guidance[range(seqs.shape[0]), first_index.squeeze()]

        if self.pref:
            mask = self.mask_token * torch.ones(guidance.shape[0], 1, device=self.device)
            mask_emb = self.item_emb(mask.long())
            guide_vector = self.pref_model(guidance, self.item_emb(guidance), mask_emb)
        else:
            #guidance = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)   # (bs, max_seq_len, hidden_size) 
            guide_vector = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)   # (bs, max_seq_len, hidden_size) 
            guide_vector = torch.sum(guide_vector, dim=1)   # (bs, hidden_size)
        
        # get the classifier label
        if self.guide_type == "item":
            first_index = -torch.sum(guidance>0, dim=-1)
            y = guidance[range(guidance.shape[0]), first_index.squeeze()]
            guide = self.item_emb(y).detach().requires_grad_(True)
        elif self.guide_type == "cond":
            guide = guide_vector.detach().requires_grad_(True)
        elif self.guide_type == "seq":
            guide = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)
            guide = guide.detach().requires_grad_(True)
        elif self.guide_type == 'bpr':
            guide = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)
            neg = torch.randint_like(guidance, 1, self.mask_token-1, device=self.dev)
            guide_neg = self.item_emb(neg.long()) * (guidance > 0).unsqueeze(-1)
            guide = (guide, guide_neg)
        else:
            raise ValueError

        xt, noise = q_xt_x0(x0, t, self.alpha_bar)  # xt: (bs, 1, seq_len, hidden_size), noise: (bs, 1, seq_len, hidden_size)
        pred_noise = self.unet(xt.squeeze().float(), t, guide_vector)
        grad = self.cond_fn(xt, guide)
        pred_noise = pred_noise - (1 - self.alpha_bar).sqrt()[0] * grad

        loss = F.mse_loss(noise.float(), pred_noise.unsqueeze(1))

        return loss
    

    def predict(self, guidance, item_indices):

        skip = self.n_steps // self.timesteps
        seq = range(0, self.n_steps, skip)

        x = torch.randn(guidance.shape[0], 1, self.seq_len, self.hidden_size).to(self.device)  # (bs, 1, seq_len, hidden_size)
        
        # get the guide vector
        if self.pref:
            mask = self.mask_token * torch.ones(guidance.shape[0], 1, device=self.device)
            mask_emb = self.item_emb(mask.long())
            guide_vector = self.pref_model(guidance, self.item_emb(guidance), mask_emb)
        else:
            guide_vector = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)
            guide_vector = torch.sum(guide_vector, dim=1)

        # get the classifier label
        if self.guide_type == "item":
            first_index = -torch.sum(guidance>0, dim=-1)
            y = guidance[range(guidance.shape[0]), first_index.squeeze()]
            guide = self.item_emb(y).detach().requires_grad_(True)
        elif self.guide_type == "cond":
            guide = guide_vector.detach().requires_grad_(True)
        elif self.guide_type == "seq":
            guide = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)
            guide = guide.detach().requires_grad_(True)
        elif self.guide_type == 'bpr':
            guide = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)
            neg = torch.randint_like(guidance, 1, self.mask_token-1, device=self.dev)
            guide_neg = self.item_emb(neg.long()) * (guidance > 0).unsqueeze(-1)
            guide = (guide, guide_neg)
        else:
            raise ValueError
        

        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            with torch.no_grad():
                pred_noise = self.unet(x.squeeze(), t, guide_vector)
                grad = self.cond_fn(x, guide)
                pred_noise = pred_noise - (1 - self.alpha_bar).sqrt()[0] * grad
                x = p_xt(x, pred_noise.unsqueeze(1), t, next_t, self.beta, self.eta)    # x: (bs, 1, seq_len, hidden_size)
        
        item_embs = self.item_emb(item_indices)
        x = x.squeeze()

        logits = []
        for i in range(self.seq_len):
            temp = x[:, i, :].unsqueeze(1).repeat(1, item_embs.shape[1], 1)
            logit = F.cosine_similarity(temp, item_embs, dim=-1)  # (bs, item_num)
            logits.append(logit)

        return logits



class DiffusionModel_CF(DiffusionModel):

    def __init__(self, user_num, item_num, device, args) -> None:

        super().__init__(user_num, item_num, device, args)

        self.mask_token = item_num + 1
        self.guidance_scale = args.classifier_scale
        self.rec_path = args.rec_path
        self.dev = device
        self.place_embedding = nn.Embedding(1, args.hidden_size)
    

    def forward(self, seqs, guidance):

        t= torch.randint(low=0, high=self.n_steps, size=(self.batch_size // 2 + 1,)).to(self.device) # bs/2+1 random select t for half batch of data
        t = torch.cat([t, self.n_steps - t - 1], dim=0)[:self.batch_size] # (bs), get the random steps for the whole batch

        x0 = self.item_emb(seqs)    # (bs, aug_len, hidden_size)
        x0 = x0.unsqueeze(1)    # (bs, 1, aug_len, hidden_size)

        if self.pref:
            mask = self.mask_token * torch.ones(guidance.shape[0], 1, device=self.device)
            mask_emb = self.item_emb(mask.long())
            guide_vector = self.pref_model(guidance, self.item_emb(guidance), mask_emb)
        else:
            guidance = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)   # (bs, max_seq_len, hidden_size) 
            guide_vector = torch.sum(guidance, dim=1)   # (bs, hidden_size)

        xt, noise = q_xt_x0(x0, t, self.alpha_bar)  # xt: (bs, 1, seq_len, hidden_size), noise: (bs, 1, seq_len, hidden_size)
        pred_noise = self.get_pred_noise(xt, t, guide_vector)

        loss = F.mse_loss(noise.float(), pred_noise.unsqueeze(1))

        return loss

    
    def predict(self, guidance, item_indices):

        skip = self.n_steps // self.timesteps
        seq = range(0, self.n_steps, skip)

        x = torch.randn(guidance.shape[0], 1, self.seq_len, self.hidden_size).to(self.device)  # (bs, 1, seq_len, hidden_size)

        # get the guide vector
        if self.pref:
            mask = self.mask_token * torch.ones(guidance.shape[0], 1, device=self.device)
            mask_emb = self.item_emb(mask.long())
            guide_vector = self.pref_model(guidance, self.item_emb(guidance), mask_emb)
        else:
            guidance = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)
            guide_vector = torch.sum(guidance, dim=1)

        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            with torch.no_grad():
                pred_noise = self.get_pred_noise(x, t, guide_vector)
                x = p_xt(x, pred_noise.unsqueeze(1), t, next_t, self.beta, self.eta)    # x: (bs, 1, seq_len, hidden_size)
        
        item_embs = self.item_emb(item_indices)
        x = x.squeeze()

        logits = []
        for i in range(self.seq_len):
            temp = x[:, i, :].unsqueeze(1).repeat(1, item_embs.shape[1], 1)
            logit = F.cosine_similarity(temp, item_embs, dim=-1)  # (bs, item_num)
            logits.append(logit)

        return logits
    
    
    # Create a classifier-free guidance sampling function
    # https://github.com/sunlin-ai/diffusion_tutorial/blob/main/diffusion_11_Classifier%20Free%20Diffusion.ipynb
    # def model_fn(self, x_t, ts, **kwargs):
    #     half = x_t[: len(x_t) // 2]
    #     combined = torch.cat([half, half], dim=0)
    #     model_out = model(combined, ts, **kwargs)
    #     eps, rest = model_out[:, :3], model_out[:, 3:]
    #     cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    #     half_eps = uncond_eps + self.guidance_scale * (cond_eps - uncond_eps)
    #     eps = torch.cat([half_eps, half_eps], dim=0)
    #     return torch.cat([eps, rest], dim=1)

    
    def get_pred_noise(self, xt, t, guide_vector):

        place_vector = self.place_embedding(torch.zeros(guide_vector.shape[0], device=self.dev).long())
        cond_noise = self.unet(xt.squeeze().float(), t, guide_vector)
        uncond_noise = self.unet(xt.squeeze().float(), t, place_vector)

        pred_noise = cond_noise + self.guidance_scale * (cond_noise - uncond_noise)

        return pred_noise
    


