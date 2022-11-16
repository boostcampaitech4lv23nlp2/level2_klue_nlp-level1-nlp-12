import torch
import pytorch_lightning as pl
import transformers
from utils import klue_re_micro_f1, klue_re_auprc, n_compute_metrics

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config.model.model_name
        self.lr = config.train.learning_rate

        # 사용할 모델을 호출합니다.
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.model_name, num_labels=30)
        # Loss 계산을 위해 사용될 CE Loss를 호출합니다.
        self.loss_func = torch.nn.CrossEntropyLoss(label_smoothing=0.2)

    def forward(self, x):
        x = self.plm(
            input_ids = x['input_ids'],
            attention_mask = x['attention_mask'],
            token_type_ids=x['token_type_ids']
        )['logits']

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)

        f1,auprc,accuracy = n_compute_metrics(logits,y).values()
        self.log("train_loss", loss)
        self.log("train_f1", f1)
        self.log("train_auprc", auprc)
        self.log("train_accuracy", accuracy)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_func(logits, y)

        f1,auprc,accuracy = n_compute_metrics(logits,y).values()
        self.log("val_loss", loss)
        self.log("val_f1", f1)
        self.log("val_auprc", auprc)
        self.log("val_accuracy", accuracy)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        f1, auprc, _ = n_compute_metrics(logits,y).values()
        self.log("test_f1", f1)
        self.log("test_auprc", auprc)

    # def predict_step(self, batch, batch_idx):
    #     x = batch
    #     logits = self(x)

    #     return logits.squeeze()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        lr_scheduler = transformers.get_constant_schedule_with_warmup(optimizer,100)
        return [optimizer], [lr_scheduler]