# -*- encoding: utf-8 -*-
# here put the import lib
from generators.generator import Generator
from generators.data import BertTrainDataset, BertRecTrainDataset, BertFinetuneDataset
from torch.utils.data import DataLoader, RandomSampler
from utils.utils import unzip_data



class BertGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):
        
        train_dataset = unzip_data(self.train, aug=self.args.aug)
        if (self.args.model_name == 'bert4rec_pretrain') | (self.args.model_name == 's3rec_pretrain'):
            train_dataset = BertTrainDataset(self.args, train_dataset, self.item_num, self.args.max_len)
        else:
            train_dataset = BertRecTrainDataset(self.args, train_dataset, self.item_num, self.args.max_len)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    
        return train_dataloader



class BertFinetuneGenerator(Generator):

    def __init__(self, args, logger, device):

        super().__init__(args, logger, device)
    

    def make_trainloader(self):
        
        train_dataset = unzip_data(self.train, aug=self.args.aug)
        train_dataset = BertFinetuneDataset(self.args, train_dataset, self.item_num, self.args.max_len)

        train_dataloader = DataLoader(train_dataset,
                                      sampler=RandomSampler(train_dataset),
                                      batch_size=self.bs,
                                      num_workers=self.num_workers)
    
        return train_dataloader








