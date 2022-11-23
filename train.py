import argparse
import os
import pickle as pickle
#############################################
import re

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertTokenizer, RobertaConfig,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer, TrainingArguments)

from load_data import *

os.chdir("/opt/ml")
wandb_dict = {
    'gwkim_22':'f631be718175f02da4e2f651225fadb8541b3cd9',
    'rion_':'0d57da7f9222522c1a3dbb645a622000e0344d36',
    'daniel0801':'b8c4d92272716adcb1b2df6597cfba448854ff90',
    'seokhee':'c79d118b300d6cff52a644b8ae6ab0933723a59f',
    'dk100':'263b9353ecef00e35bdf063a51a82183544958cc'
}

time_ = datetime.now() + timedelta(hours=9)
time_now = time_.strftime("%m%d%H%M")
#############################################

def klue_re_micro_f1(preds, labels):
    """KLUE-RE micro f1 (except no_relation)"""
    label_list = [
        "no_relation",
        "org:top_members/employees",
        "org:members",
        "org:product",
        "per:title",
        "org:alternate_names",
        "per:employee_of",
        "org:place_of_headquarters",
        "per:product",
        "org:number_of_employees/members",
        "per:children",
        "per:place_of_residence",
        "per:alternate_names",
        "per:other_family",
        "per:colleagues",
        "per:origin",
        "per:siblings",
        "per:spouse",
        "org:founded",
        "org:political/religious_affiliation",
        "org:member_of",
        "per:parents",
        "org:dissolved",
        "per:schools_attended",
        "per:date_of_death",
        "per:date_of_birth",
        "per:place_of_birth",
        "per:place_of_death",
        "org:founded_by",
        "per:religion",
    ]
    no_relation_label_idx = label_list.index("no_relation")
    label_indices = list(range(len(label_list)))
    label_indices.remove(no_relation_label_idx)
    return (
        sklearn.metrics.f1_score(labels, preds, average="micro", labels=label_indices)
        * 100.0
    )



def klue_re_auprc(probs, labels):
    """KLUE-RE AUPRC (with no_relation)"""
    labels = np.eye(30)[labels]

    score = np.zeros((30,))
    for c in range(30):
        targets_c = labels.take([c], axis=1).ravel()
        preds_c = probs.take([c], axis=1).ravel()
        precision, recall, _ = sklearn.metrics.precision_recall_curve(
            targets_c, preds_c
        )
        score[c] = sklearn.metrics.auc(recall, precision)
    return np.average(score) * 100.0


def compute_metrics(pred):
    """validation을 위한 metrics function"""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    probs = pred.predictions

    # calculate accuracy using sklearn's function
    f1 = klue_re_micro_f1(preds, labels)
    auprc = klue_re_auprc(probs, labels)
    acc = accuracy_score(labels, preds)  # 리더보드 평가에는 포함되지 않습니다.

    return {
        "micro f1 score": f1,
        "auprc": auprc,
        "accuracy": acc,
    }


def label_to_num(label):
    num_label = []
    with open("dict_label_to_num.pkl", "rb") as f:
        dict_label_to_num = pickle.load(f)
    for v in label:
        num_label.append(dict_label_to_num[v])

    return num_label


def train():
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = "klue/bert-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("../dataset/train/train.csv")
    # dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset["label"].values)
    # dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    # tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    # RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(device)
    # setting model hyperparameter
    model_config = AutoConfig.from_pretrained(MODEL_NAME)
    model_config.num_labels = 30

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=model_config
    )
    print(model.config)
    model.parameters
    model.to(device)

    # 사용한 option 외에도 다양한 option들이 있습니다.
    # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
        output_dir="./results",  # output directory
        save_total_limit=5,  # number of total save model.
        save_steps=500,  # model saving step.
        num_train_epochs=20,  # total number of training epochs
        learning_rate=5e-5,  # learning_rate
        per_device_train_batch_size=16,  # batch size per device during training
        per_device_eval_batch_size=16,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir="./logs",  # directory for storing logs
        logging_steps=100,  # log saving step.
        evaluation_strategy="steps",  # evaluation strategy to adopt during training
        # `no`: No evaluation during training.
        # `steps`: Evaluate every `eval_steps`.
        # `epoch`: Evaluate every end of epoch.
        eval_steps=500,  # evaluation step.
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_train_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
    )

    # train model
    trainer.train()
    model.save_pretrained("./best_model")


# def main():
#     train()


if __name__ == "__main__":
    # main()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='base_config')
    args, _ = parser.parse_known_args()
    
    cfg = OmegaConf.load(f'./config/{args.config}.yaml')

    #os.environ["WANDB_API_KEY"] = wandb_dict[cfg.wandb.wandb_username]
    wandb.login(key = wandb_dict[cfg.wandb.wandb_username])
    model_name_ch = re.sub('/','_',cfg.model.model_name)
    wandb_logger = WandbLogger(
                log_model="all",
                name=f'{model_name_ch}_{cfg.train.batch_size}_{cfg.train.learning_rate}_{time_now}',
                project=cfg.wandb.wandb_project,
                entity=cfg.wandb.wandb_entity
                )

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_pearson',
                                        save_top_k=1,
                                        save_last=True,
                                        save_weights_only=False,
                                        verbose=False,
                                        mode='max')

    # Earlystopping
    earlystopping = EarlyStopping(monitor='val_pearson', patience=2, mode='max')
    
    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(cfg.model.model_name, cfg.train.batch_size, cfg.data.shuffle, cfg.path.train_path, cfg.path.dev_path,
                            cfg.path.test_path, cfg.path.predict_path)
    model = Model(cfg)

    # training_args = TrainingArguments(seed=args.train.seed)
    # gpu가 없으면 'gpus=0'을, gpu가 여러개면 'gpus=4'처럼 사용하실 gpu의 개수를 입력해주세요
    trainer = pl.Trainer(
        gpus=-1, 
        max_epochs=cfg.train.max_epoch, 
        log_every_n_steps=cfg.train.logging_step,
        logger=wandb_logger,    # W&B integration
        callbacks = [checkpoint_callback, earlystopping]
        )
    # Train part
    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader)

    # 학습이 완료된 모델을 저장합니다.
    output_dir_path = 'output'
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_path = os.path.join(output_dir_path, f'{model_name_ch}_{time_now}_model.pt')
    torch.save(model, output_path)
def main():
    train()


if __name__ == "__main__":
    main()
