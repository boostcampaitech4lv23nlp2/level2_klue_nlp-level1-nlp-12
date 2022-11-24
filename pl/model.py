import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional
import transformers


warnings.filterwarnings(action="ignore")

from importlib import import_module

from utils import criterion_entrypoint, klue_re_auprc, klue_re_micro_f1, n_compute_metrics


class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config.model.model_name
        self.lr = config.train.learning_rate
        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=30
        )
        # Loss 계산을 위해 사용될 CE Loss를 호출합니다.
        self.loss_func = criterion_entrypoint(config.train.loss_name)
        self.optimizer_name = config.train.optimizer_name

    def forward(self, x):
        x = self.plm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            token_type_ids=x["token_type_ids"],
            labels=x["labels"],
        )
        return x["logits"]

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)
        loss = self.loss_func(logits, y.long())

        f1, auprc, accuracy = n_compute_metrics(logits, y).values()
        self.log("train_loss", loss)
        self.log("train_f1", f1)
        self.log("train_auprc", auprc)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)
        loss = self.loss_func(logits, y.long())

        f1, auprc, accuracy = n_compute_metrics(logits, y).values()
        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_f1", f1, on_step=True)
        self.log("val_auprc", auprc, on_step=True)
        self.log("val_accuracy", accuracy, on_step=True)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)

        f1, auprc, _ = n_compute_metrics(logits, y).values()
        self.log("test_f1", f1)
        self.log("test_auprc", auprc)

    def configure_optimizers(self):
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        opt_module = getattr(import_module("torch.optim"), self.optimizer_name)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            # weight_decay=5e-4
        )
        return optimizer
