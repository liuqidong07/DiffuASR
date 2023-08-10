# -*- encoding: utf-8 -*-
# here put the import lib
import os
import time
import torch
import numpy as np
from tqdm import tqdm
from trainers.trainer import Trainer
from utils.utils import metric_report, metric_len_report, record_csv


class SeqTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            #neg = neg.view(-1) # (bs, neg_num) --> (bs * neg_num)
            #pos_logits, neg_logits = self.model(seq, pos, neg)  # seq: (bs, max_len)
            #pos_labels, neg_labels = torch.ones(pos_logits.shape, device=self.device), torch.zeros(neg_logits.shape, device=self.device)
            #neg_logits = neg_logits.view(pos.shape[0], -1)  # (bs * neg_num) --> (bs, neg_num)
            #indices = (pos != 0)    # do not calculate the padding units
            #pos_loss, neg_loss = self.loss_func(pos_logits[indices], pos_labels[indices]), self.loss_func(neg_logits[indices], neg_labels[indices])
            #loss = pos_loss + neg_loss
            loss = self.model(seq, pos, neg, positions)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



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
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            seq, pos, neg, positions = batch
            seq_len = torch.cat([seq_len, torch.sum(seq>0, dim=1)])
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            
            with torch.no_grad():

                #neg = neg.view(-1) # (bs, neg_num) --> (bs * neg_num)
                #pos_logits, neg_logits = self.model(seq, pos, neg)
                #neg_logits = neg_logits.view(pos.shape[0], -1)  # (bs * neg_num) --> (bs, neg_num)
                #pred_logits = -torch.cat([pos_logits, neg_logits], dim=1)
                pred_logits = -self.model.predict(seq, torch.cat([pos.unsqueeze(1), neg], dim=1), positions)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len)
        
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        for k, v in res_len_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict



class CL4SRecTrainer(SeqTrainer):

    def __init__(self, args, logger, writer, device, generator):
        
        super().__init__(args, logger, writer, device, generator)


    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq, pos, neg, positions, aug1, aug2 = batch
            seq, pos, neg, positions, aug1, aug2 = seq.long(), pos.long(), neg.long(), positions.long(), aug1.long(), aug2.long()
            aug = (aug1, aug2)
            loss = self.model(seq, pos, neg, positions, aug)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



class SSEPTTrainer(Trainer):

    def __init__(self, args, logger, writer, device, generator):

        super().__init__(args, logger, writer, device, generator)
    

    def _train_one_epoch(self, epoch):

        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_loader, leave=False, desc='Training')

        for batch in prog_iter:

            batch = tuple(t.to(self.device) for t in batch)

            train_start = time.time()
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            loss = self.model(seq_user, pos_user, neg_user, seq, pos, neg, positions)
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += 1
            nb_tr_steps += 1

            # Display loss
            prog_iter.set_postfix(loss='%.4f' % (tr_loss / nb_tr_steps))

            self.optimizer.step()
            self.optimizer.zero_grad()

            train_end = time.time()
            train_time.append(train_end-train_start)

        self.writer.add_scalar('train/loss', tr_loss / nb_tr_steps, epoch)



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
        pred_rank = torch.empty(0).to(self.device)
        seq_len = torch.empty(0).to(self.device)

        for batch in tqdm(test_loader, desc=desc):

            batch = tuple(t.to(self.device) for t in batch)
            seq_user, pos_user, neg_user, seq, pos, neg, positions = batch
            seq, pos, neg, positions = seq.long(), pos.long(), neg.long(), positions.long()
            seq_user, pos_user, neg_user = seq_user.long(), pos_user.long(), neg_user.long()
            seq_len = torch.cat([seq_len, torch.sum(seq>0, dim=1)])

            with torch.no_grad():

                pred_logits = -self.model.predict(seq_user, seq, torch.cat([pos_user.unsqueeze(1), neg_user], dim=1), torch.cat([pos.unsqueeze(1), neg], dim=1), positions)

                per_pred_rank = torch.argsort(torch.argsort(pred_logits))[:, 0]
                pred_rank = torch.cat([pred_rank, per_pred_rank])

        self.logger.info('')
        res_dict = metric_report(pred_rank.detach().cpu().numpy())
        res_len_dict = metric_len_report(pred_rank.detach().cpu().numpy(), seq_len.detach().cpu().numpy(), aug_len=self.args.aug_seq_len)
        
        for k, v in res_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        for k, v in res_len_dict.items():
            if not test:
                self.writer.add_scalar('Test/{}'.format(k), v, epoch)
            self.logger.info('%s: %.5f' % (k, v))
        
        res_dict = {**res_dict, **res_len_dict}

        if test:
            record_csv(self.args, res_dict)
        
        return res_dict
