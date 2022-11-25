import pytorch_lightning as pl
import torch
import torch.nn.functional
import transformers
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

        self.log("train_loss", loss)
        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("train_f1", f1)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)
        loss = self.loss_func(logits, y.long())

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("val_f1", f1, on_step=True)
        self.log("val_accuracy", accuracy, on_step=True)

        return {"logits": logits, "y": y}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu()

        auprc = klue_re_auprc(logits, y)
        self.log("val_auprc", auprc, on_step=True)

    def test_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)

        f1, auprc, _ = n_compute_metrics(logits, y).values()
        self.log("test_f1", f1)
        self.log("test_auprc", auprc)

    def configure_optimizers(self):
        opt_module = getattr(import_module("torch.optim"), self.optimizer_name)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            # weight_decay=5e-4
        )
        if self.lr_sch_use:
            _scheduler_dic = {
                "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, self.lr_decay_step, gamma=0.5),
                "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
                "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0.0),
            }
            scheduler = _scheduler_dic[self.scheduler_name]
            return [optimizer], [scheduler]
        else:
            return optimizer