import argparse
import os
import re
import warnings

from datetime import datetime, timedelta

import wandb

from data import *
from model import *
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

warnings.filterwarnings(action="ignore")

time_ = datetime.now() + timedelta(hours=9)
time_now = time_.strftime("%m%d%H%M")

wandb_dict = {
    "user": "token"
}

if __name__ == "__main__":
    # 하이퍼 파라미터 등 각종 설정값을 입력받습니다
    # 터미널 실행 예시 : python3 run.py --batch_size=64 ...
    # 실행 시 '--batch_size=64' 같은 인자를 입력하지 않으면 default 값이 기본으로 실행됩니다
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f"./config/{args.config}.yaml")
    pl.seed_everything(cfg.train.seed, workers=True)

    # os.environ["WANDB_API_KEY"] = wandb_dict[cfg.wandb.wandb_username]
    wandb.login(key=wandb_dict[cfg.wandb.wandb_username])
    model_name_ch = re.sub("/", "_", cfg.model.model_name)
    wandb_logger = WandbLogger(
        log_model="all",
        name=f"{cfg.model.saved_name}_{cfg.train.batch_size}_{cfg.train.learning_rate}_{time_now}",
        project=cfg.wandb.wandb_project,
        entity=cfg.wandb.wandb_entity,
    )

    ck_dir_path = f"/opt/ml/code/pl/checkpoint/{model_name_ch}"
    if not os.path.exists(ck_dir_path):
        os.makedirs(ck_dir_path)

    # Checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=ck_dir_path,
        filename="{epoch}_{val_f1:.2f}",
        monitor="val_f1",
        save_top_k=1,
        mode="max",
    )

    # Earlystopping
    earlystopping = EarlyStopping(monitor="val_f1", patience=2, mode="max")

    model = Model(cfg)
    results = []

    for k in range(cfg.train.nums_folds):

        datamodule = Dataloader(
            cfg.model.model_name,
            cfg.train.batch_size,
            cfg.data.shuffle,
            cfg.path.train_path,
            cfg.path.test_path,
            k=k,
            split_seed=cfg.train.seed,
            num_splits=cfg.train.nums_folds,
        )

        trainer = pl.Trainer(
            precision=16,
            accelerator="gpu",
            devices=1,
            max_epochs=cfg.train.max_epoch,
            log_every_n_steps=cfg.train.logging_step,
            logger=wandb_logger,
            callbacks=[checkpoint_callback, earlystopping, RichProgressBar()],
            deterministic=True,
            # limit_train_batches=0.05,
        )
        trainer.fit(model=model, datamodule=datamodule)
        score = trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

        results.extend(score)

    # Fold 적용 결과 확인
    show_result(results)

    # 학습이 완료된 모델을 저장합니다.
    output_dir_path = "/opt/ml/code/pl/output"
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    output_path = os.path.join(output_dir_path, f"{model_name_ch}_{time_now}_model.pt")
    torch.save(model.state_dict(), output_path)
