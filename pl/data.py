import os
import pickle as pickle
import pandas as pd
import torch
from tqdm.auto import tqdm
import transformers
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from utils import *

class Dataset(torch.utils.data.Dataset):
    """ Dataset 구성을 위한 Class """
    def __init__(self,pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self,idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)

class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, test_path, val_ration=0.3):
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.test_path = test_path
        self.val_ration = val_ration

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=160)

    def setup(self, stage='fit'):
        if stage == 'fit':
            # 학습 데이터을 호출
            total_data = pd.read_csv(self.train_path)
            total_dataset = preprocessing_datset(train_data)
            total_label = label_to_num(total_dataset['label'].values)
            tokenized_total = tokenized_dataset(total_dataset, self.tokenizer)

            #KFold 
            kf = KFold()
