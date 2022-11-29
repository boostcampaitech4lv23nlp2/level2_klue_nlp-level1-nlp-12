import pickle as pickle

import pandas as pd
import pytorch_lightning as pl
import torch
import transformers

from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm
from utils import *

class Dataset(torch.utils.data.Dataset):
    """Dataset 구성을 위한 Class"""

    def __init__(self, pair_dataset, labels):
        self.pair_dataset = pair_dataset
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.pair_dataset.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class Dataloader(pl.LightningDataModule):
    def __init__(
        self,
        model_name,
        batch_size,
        shuffle,
        train_path,
        test_path,
        k=1,
        split_seed=42,
        num_splits=5,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.k = k
        self.split_seed = split_seed
        self.num_splits = num_splits

        self.train_path = train_path
        self.test_path = test_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, max_length=200)

    def setup(self, stage="fit"):
        if stage == "fit":
            # StratifiedKFold
            kf = StratifiedKFold(
                n_splits=self.num_splits,
                shuffle=True,
                random_state=self.split_seed,
            )
            # 학습 데이터을 호출
            total_data = load_data(self.train_path)
            total_label = label_to_num(total_data["label"].values)
            tokenized_total = tokenized_dataset(total_data, self.tokenizer)
            total_dataset = Dataset(tokenized_total, total_label)

            all_splits = [k for k in kf.split(total_dataset,total_label)]
            # k번째 Fold Dataset 선택
            train_indexes, val_indexes = all_splits[self.k]
            train_indexes, val_indexes = train_indexes.tolist(), val_indexes.tolist()
            # fold한 index에 따라 데이터셋 분할
            self.train_dataset = [total_dataset[x] for x in train_indexes]
            self.val_dataset = [total_dataset[x] for x in val_indexes]

        else:
            total_data = load_data(self.train_path)

            train_data = total_data.sample(frac=0.9, random_state=self.split_seed)
            val_data = total_data.drop(train_data.index)

            val_label = label_to_num(val_data["label"].values)
            tokenized_val = tokenized_dataset(val_data, self.tokenizer)

            self.test_dataset = Dataset(tokenized_val, val_label)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)
