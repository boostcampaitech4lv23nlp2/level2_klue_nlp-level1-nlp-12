import warnings

import pytorch_lightning as pl
import torch
import torch.nn.functional
import transformers

warnings.filterwarnings(action="ignore")

from utils import klue_re_auprc, klue_re_micro_f1, n_compute_metrics


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
        # self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.2)
        self.loss_func = torch.nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids, labels):
        x = self.plm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        return x["logits"]

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        y = batch["labels"]

        logits = self(input_ids, attention_mask, token_type_ids, y)
        loss = self.loss_func(logits, y)

        f1, auprc, accuracy = n_compute_metrics(logits, y).values()
        self.log("train_loss", loss)
        self.log("train_f1", f1)
        self.log("train_auprc", auprc)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        y = batch["labels"]

        logits = self(input_ids, attention_mask, token_type_ids, y)
        loss = self.loss_func(logits, y)

        f1, auprc, accuracy = n_compute_metrics(logits, y).values()
        self.log("val_loss", loss)
        self.log("val_f1", f1)
        self.log("val_auprc", auprc)
        self.log("val_accuracy", accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        token_type_ids = batch["token_type_ids"]
        y = batch["labels"]

        logits = self(input_ids, attention_mask, token_type_ids, y)

        f1, auprc, _ = n_compute_metrics(logits, y).values()
        self.log("test_f1", f1)
        self.log("test_auprc", auprc)

    # def predict_step(self, batch, batch_idx):
    #     x = batch
    #     logits = self(x)

    #     return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        lr_scheduler = transformers.get_constant_schedule_with_warmup(optimizer, 100)
        return [optimizer], [lr_scheduler]
