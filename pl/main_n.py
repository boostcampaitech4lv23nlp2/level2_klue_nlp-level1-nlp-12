import argparse
import os
import re

from datetime import datetime, timedelta

import torch
import wandb

from data_n import *
from model import *
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from data_n import *
from model import *
from transformers import TrainingArguments

time_ = datetime.now() + timedelta(hours=9)
time_now = time_.strftime("%m%d%H%M")

wandb_dict = {
    "gwkim_22": "f631be718175f02da4e2f651225fadb8541b3cd9",
    "rion_": "0d57da7f9222522c1a3dbb645a622000e0344d36",
    "daniel0801": "b8c4d92272716adcb1b2df6597cfba448854ff90",
    "seokhee": "c79d118b300d6cff52a644b8ae6ab0933723a59f",
    "dk100": "263b9353ecef00e35bdf063a51a82183544958cc",
}


class MyTrainer(pl.Trainer):
    # loss_name 이라는 인자를 추가로 받아 self에 각인 시켜줍니다.
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name = loss_name  # 각인!

    def compute_loss(self, model, inputs, return_outputs=False):
        # custom_loss = criterion_entrypoint(self.loss_name)
        if self.loss_name == "CrossEntropy":
            # lossname이 CrossEntropy 이면, custom_loss에 torch.nn.CrossEntropyLoss()를 선언(?) 해줍니다.
            custom_loss = torch.nn.MSELoss()

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)

        if labels is not None:
            # loss를 계산 하던 부분에 custom_loss를 이용해 계산하는 코드를 넣기
            loss = custom_loss(outputs[0], labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f"/opt/ml/code/pl/config/{args.config}.yaml")

    # os.environ["WANDB_API_KEY"] = wandb_dict[cfg.wandb.wandb_username]
    wandb.login(key=wandb_dict[cfg.wandb.wandb_username])
    model_name_ch = re.sub("/", "_", cfg.model.model_name)
    wandb_logger = WandbLogger(
        log_model="all",
        name=f"{cfg.model.saved_name}_{cfg.train.batch_size}_{cfg.train.learning_rate}_{time_now}",
        project=cfg.wandb.wandb_project,
        entity=cfg.wandb.wandb_entity,
    )

    pl.seed_everything(cfg.train.seed, workers=True)

    ck_dir_path = f"/opt/ml/code/pl/checkpoint/{model_name_ch}"
    if not os.path.exists(ck_dir_path):
        os.makedirs(ck_dir_path)

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath="/opt/ml/code/level2_klue_nlp-level1-nlp-12/pl/checkpoint",
        auto_insert_metric_name=True,
        monitor="val_loss",
        save_top_k=1,
        mode="min",
    )

    # Earlystopping
    earlystopping = EarlyStopping(monitor="val_f1", patience=2, mode="min")

    # dataloader와 model을 생성합니다.
    dataloader = Dataloader(
        cfg.model.model_name,
        cfg.train.batch_size,
        cfg.data.shuffle,
        cfg.path.train_path,
        cfg.path.test_path,
        cfg.train.seed,
    )
    model = Model(cfg)

    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        devices=1,
        max_epochs=cfg.train.max_epoch,
        log_every_n_steps=cfg.train.logging_step,
        logger=wandb_logger,  # W&B integration
        callbacks=[earlystopping, checkpoint_callback, RichProgressBar()],
        deterministic=True,
        # limit_train_batches=0.15,  # use only 30% of training data
        # limit_val_batches = 0.01, # use only 1% of val data
        # limit_train_batches=10    # use only 10 batches of training data
    )
    # trainer = MyTrainer(
    #     accelerator="gpu",
    #     devices=1,
    #     max_epochs=cfg.train.max_epoch,
    #     log_every_n_steps=cfg.train.logging_step,
    #     logger=wandb_logger,  # W&B integration
    #     callbacks=[
    #         earlystopping,
    #     ],
    #     deterministic=True,
    #     loss_name="CrossEntropy",  # CrossEntropy, focal, label_smoothing, f1
    # )

    trainer.fit(model=model, datamodule=dataloader)
    trainer.test(model=model, datamodule=dataloader, ckpt_path="best")

    # 학습이 완료된 모델을 저장합니다.
    output_dir_path = "output"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_path = os.path.join(output_dir_path, f"{model_name_ch}_{time_now}_model.pt")
    torch.save(model.state_dict(), output_path)
