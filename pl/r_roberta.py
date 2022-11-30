from importlib import import_module

import numpy as np
import pytorch_lightning as pl
import torch
import transformers

from utils import criterion_entrypoint, klue_re_auprc, klue_re_micro_f1, n_compute_metrics

class FCLayer(pl.LightningModule):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super().__init__()
        self.save_hyperparameters()

        self.use_activation = use_activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.tanh = torch.nn.Tanh()

        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.model_name = config.model.model_name
        self.lr = config.train.learning_rate
        self.lr_sch_use = config.train.lr_sch_use
        self.lr_decay_step = config.train.lr_decay_step
        self.scheduler_name = config.train.scheduler_name
        self.lr_weight_decay = config.train.lr_weight_decay
        self.dr_rate = 0
        self.hidden_size = 1024
        self.num_classes = 30

        # 사용할 모델을 호출합니다.
        self.plm = transformers.RobertaModel.from_pretrained(self.model_name,add_pooling_layer=False)
        self.cls_fc = FCLayer(self.hidden_size, self.hidden_size//2, self.dr_rate)
        self.sentence_fc = FCLayer(self.hidden_size, self.hidden_size//2, self.dr_rate)
        self.label_classifier = FCLayer(self.hidden_size//2 * 3, self.num_classes, self.dr_rate, False)

        # Loss 계산을 위해 사용될 CE Loss를 호출합니다.
        self.loss_func = criterion_entrypoint(config.train.loss_name)
        self.optimizer_name = config.train.optimizer_name

    def forward(self,x):
        out = self.plm(
            input_ids=x["input_ids"],
            attention_mask=x["attention_mask"],
            token_type_ids=x["token_type_ids"],
        )[0]

        sentence_end_position = torch.where(x["input_ids"] == 2)[1]
        sent1_end, sent2_end = sentence_end_position[0], sentence_end_position[1]
        
        cls_vector = out[:, 0, :] # take <s> token (equiv. to [CLS])
        prem_vector = out[:,1:sent1_end]              # Get Premise vector
        hypo_vector = out[:,sent1_end+1:sent2_end]    # Get Hypothesis vector

        prem_vector = torch.mean(prem_vector, dim=1) # Average
        hypo_vector = torch.mean(hypo_vector, dim=1)
        
        # Dropout -> tanh -> fc_layer (Share FC layer for premise and hypothesis)
        cls_embedding = self.cls_fc(cls_vector)
        prem_embedding = self.sentence_fc(prem_vector)
        hypo_embedding = self.sentence_fc(hypo_vector)
        
        # Concat -> fc_layer
        concat_embedding = torch.cat([cls_embedding, prem_embedding, hypo_embedding], dim=-1)
        
        return self.label_classifier(concat_embedding)

    def training_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)
        loss = self.loss_func(logits, y.long())

        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("train", {"loss": loss, "f1": f1, "accuracy": accuracy})

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)
        loss = self.loss_func(logits, y.long())

        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("val_loss", loss)
        self.log("val_accuracy", accuracy)
        self.log("val_f1", f1, on_step=True)

        return {"logits": logits, "y": y}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu()

        auprc = klue_re_auprc(logits, y)
        self.log("val_auprc", auprc)

    def test_step(self, batch, batch_idx):
        x = batch
        y = batch["labels"]

        logits = self(x)

        f1, accuracy = n_compute_metrics(logits, y).values()
        self.log("test_f1", f1)

        return {"logits": logits, "y": y}

    def test_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs])
        y = torch.cat([x["y"] for x in outputs])

        logits = logits.detach().cpu().numpy()
        y = y.detach().cpu()

        auprc = klue_re_auprc(logits, y)
        self.log("test_auprc", auprc)

    def predict_step(self, batch, batch_idx):
        logits = self(batch)

        return logits

    def configure_optimizers(self):
        opt_module = getattr(import_module("torch.optim"), self.optimizer_name)
        if self.lr_weight_decay:
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                weight_decay=0.01
            )
        else:
            optimizer = opt_module(
                filter(lambda p: p.requires_grad, self.parameters()),
                lr=self.lr,
                # weight_decay=5e-4
            )
        if self.lr_sch_use:
            t_total = 2030 * 7 # train_dataloader len, epochs
            warmup_step = int(t_total * 0.1)

            _scheduler_dic = {
                "StepLR": torch.optim.lr_scheduler.StepLR(optimizer, self.lr_decay_step, gamma=0.5),
                "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10),
                "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2, eta_min=0.0),
                "constant_warmup": transformers.get_constant_schedule_with_warmup(optimizer, 100),
                "cosine_warmup" : transformers.get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=10, num_training_steps=t_total)
            }
            scheduler = _scheduler_dic[self.scheduler_name]
            
            return [optimizer], [scheduler]
        else:
            return optimizer
