# -*- encoding: utf-8 -*-
'''
@File    :   diffusion_trainer.py
@Time    :   2023/02/21 09:53:13
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import os
import numpy as np
import torch
from tqdm import tqdm, trange
from trainers.trainer import Trainer
from utils.utils import load_pretrained_model, seq_acc
from diffusion.diffusion import DiffusionModel, DiffusionModel_CG, DiffusionModel_CF
from diffusion.diffusionlm import DiffusionLM
from diffusion.ema import EMAHelper



class DiffusionTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)

        if args.ema:

            self.ema_helper = EMAHelper(mu=args.ema_rate)
            self.ema_helper.register(self.model)

        if args.pretrain_item:

            self._load_rec_model(args.rec_path)

        if args.freeze_item:

            self._freeze_item()

    
    def _create_model(self):
        
        if self.args.model_name == 'diffusion':
            if self.args.guide_model == "none":
                self.model = DiffusionModel(self.user_num, self.item_num, self.device, self.args)
            elif self.args.guide_model == "cg":
                self.model = DiffusionModel_CG(self.user_num, self.item_num, self.device, self.args)
            elif self.args.guide_model == "cf":
                self.model = DiffusionModel_CF(self.user_num, self.item_num, self.device, self.args)
        elif self.args.model_name == "diffusionlm":
            self.model = DiffusionLM(self.user_num, self.item_num, self.device, self.args)
        else:
            raise NotImplementedError
        
        self.model.to(self.device)

    
    def load_model(self):

        self.model = load_pretrained_model(self.args.pretrain_dir, self.model, self.logger, device=self.device)

    
    def _load_rec_model(self, rec_path):

        self.logger.info("Loading recommendation model ... ")
        checkpoint_path = os.path.join(rec_path, 'pytorch_model.bin')

        model_dict = self.model.state_dict()
        try:
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device)['state_dict']
        except:
            pretrained_dict = torch.load(checkpoint_path, map_location=self.device)

        # filter out required parameters
        #required_params = ['item_emb']
        new_dict = {k: v for k, v in pretrained_dict.items() if "item_emb" in k}
        model_dict.update(new_dict)
        self.logger.info('Total loaded parameters: {}, update: {}'.format(len(pretrained_dict), len(new_dict)))
        self.model.load_state_dict(model_dict)

    
    def _freeze_item(self):

        for name, param in self.model.named_parameters():
    
            if 'item_emb' in name:

                param.requires_grad = False


    def train(self):

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Running training **********")
        self.logger.info("  Batch size = %d", self.args.train_batch_size)

        for epoch in trange(int(self.args.num_train_epochs), desc="Epoch"):

            avg_loss = self._train_one_epoch(epoch)

            if (epoch+1)%100 == 0:

                self.eval(epoch)

            self.stopper(-avg_loss, epoch, model_to_save, self.optimizer, self.scheduler)

            if self.stopper.early_stop:

                break
        
        best_epoch = self.stopper.best_epoch
        self.logger.info('')
        self.logger.info('The best epoch is %d' % best_epoch)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')
        loss_list = []

        self.logger.info("\n----------------------------------------------------------------")
        self.logger.info("********** Epoch: %d Train **********" % epoch)

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            seq, positions, diff_seq = batch
            seq, positions, diff_seq = seq.long(), positions.long(), diff_seq.long()
            loss = self.model(diff_seq, seq)
            loss_list.append(loss.item())
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.args.ema:

                self.ema_helper.update(self.model)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)

        self.logger.info("The loss value %.5f" % np.mean(loss_list))

        return np.mean(loss_list)


    def augment(self):

        self.model.eval()
        aug_data = []
        aug_loader = self.generator.make_augmentloader()

        for batch in tqdm(aug_loader):

            batch = tuple(t.to(self.device) for t in batch)
            seq, positions = batch
            seq, positions = seq.long(), positions.long()
            item_indicies = torch.arange(1, self.item_num+1)    # (1, item_num) to get the item embedding matrix
            item_indicies = item_indicies.repeat(seq.shape[0], 1)   # (bs, item_num)
            item_indicies = item_indicies.to(self.device).long()
            per_aug_data = torch.empty(0).to(self.device)

            with torch.no_grad():

                logits = self.model.predict(seq, item_indicies)

            for i in range(self.args.aug_num):
                
                if self.args.model_name == "diffusionlm":
                    logit = logits[i]
                    aug_item = torch.topk(logit, k=1, dim=-1).indices
                    aug_item = aug_item.squeeze(-1)
                else:
                    logit = logits[i]
                    aug_item = torch.argsort(logit, descending=True)[:, 0]   # return the index of max score
                    aug_item = aug_item + 1
                    aug_item = aug_item.unsqueeze(1)    # (bs, 1)
                per_aug_data = torch.cat([per_aug_data, aug_item], dim=1)  # [..., n-3, n-2, n-1]

            aug_data.append(per_aug_data)

        aug_data = torch.cat(aug_data, dim=0)

        aug_data = aug_data.detach().cpu().numpy()
        self.generator.save_aug(aug_data)
    

    def eval(self, epoch=0, test=False):

        print('')
        if test:
            self.logger.info("\n----------------------------------------------------------------")
            self.logger.info("********** Running test **********")
            desc = 'Testing'
            model_state_dict = torch.load(os.path.join(self.args.output_dir, 'pytorch_model.bin'))
            try:
                self.model.load_state_dict(model_state_dict['state_dict'])
            except:
                self.model.load_state_dict(model_state_dict)
            self.model.to(self.device)
            test_loader = self.test_loader
        
        else:
            self.logger.info("\n----------------------------------")
            self.logger.info("********** Epoch: %d eval **********" % epoch)
            desc = 'Evaluating'
            test_loader = self.valid_loader
        
        self.model.eval()
        true_data = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)
        pred_data = []

        for batch in tqdm(test_loader):

            batch = tuple(t.to(self.device) for t in batch)
            seq, positions, true = batch
            seq, positions = seq.long(), positions.long()
            item_indicies = torch.arange(1, self.item_num+1)    # (1, item_num) to get the item embedding matrix
            item_indicies = item_indicies.repeat(seq.shape[0], 1)   # (bs, item_num)
            item_indicies = item_indicies.to(self.device).long()
            per_pred_data = torch.empty(0).to(self.device)
            true_data = torch.cat([true_data, true], dim=0)

            with torch.no_grad():

                logits = self.model.predict(seq, item_indicies)

            for i in range(self.args.aug_num):
                
                if self.args.model_name == "diffusionlm":
                    logit = logits[i]
                    aug_item = torch.topk(logit, k=1, dim=-1).indices
                    aug_item = aug_item.squeeze(-1)
                else:
                    logit = logits[i]
                    aug_item = torch.argsort(logit, descending=True)[:, 0]   # return the index of max score
                    aug_item = aug_item + 1
                    aug_item = aug_item.unsqueeze(1)    # (bs, 1)
                per_pred_data = torch.cat([per_pred_data, aug_item], dim=1)  # [..., n-3, n-2, n-1]
            
            pred_data.append(per_pred_data)
        
        pred_data = torch.cat(pred_data, dim=0)

        self.logger.info('')
        res_dict = seq_acc(true_data.detach().cpu().numpy(), pred_data.long().detach().cpu().numpy())
        
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        
        return res_dict
        

