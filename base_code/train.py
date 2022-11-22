from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, BertTokenizer, RobertaConfig,
                          RobertaForSequenceClassification, RobertaTokenizer,
                          Trainer, TrainingArguments)

from data import *
from utils import *
import torch.nn as nn
from utils import criterion_entrypoint

class MyTrainer(Trainer):
    # loss_name 이라는 인자를 추가로 받아 self에 각인 시켜줍니다.
    def __init__(self, loss_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name= loss_name # 각인!

    def compute_loss(self, model, inputs, return_outputs=False):

        # config에 저장된 loss_name에 따라 다른 loss 계산 
        # if self.loss_name == 'CrossEntropy':
        #     # lossname이 CrossEntropy 이면, custom_loss에 torch.nn.CrossEntropyLoss()를 선언(?) 해줍니다.
        #     custom_loss = torch.nn.CrossEntropyLoss()
        custom_loss = criterion_entrypoint(self.loss_name)
                        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        outputs = model(**inputs)

        if labels is not None:
            #loss를 계산 하던 부분에 custom_loss를 이용해 계산하는 코드를 넣어 줍니다!
            #원본 코드를 보시면 output[0]가 logit 임을 알 수 있습니다!
            loss = custom_loss(outputs[0], labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss


def train(cfg):
    # load model and tokenizer
    # MODEL_NAME = "bert-base-uncased"
    MODEL_NAME = cfg.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # load dataset
    train_dataset = load_data("../dataset/train/train.csv")
    dev_dataset = load_data("../dataset/train/dev.csv") # validation용 데이터는 따로 만드셔야 합니다.

    train_label = label_to_num(train_dataset["label"].values)
    dev_label = label_to_num(dev_dataset['label'].values)

    # tokenizing dataset
    tokenized_train = tokenized_dataset(train_dataset, tokenizer)
    tokenized_dev = tokenized_dataset(dev_dataset, tokenizer)

    # make dataset for pytorch.
    RE_train_dataset = RE_Dataset(tokenized_train, train_label)
    RE_dev_dataset = RE_Dataset(tokenized_dev, dev_label)

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
        report_to="wandb",
    )

    # trainer = Trainer(
        # model=model,  # the instantiated 🤗 Transformers model to be trained
        # args=training_args,  # training arguments, defined above
        # train_dataset=RE_train_dataset,  # training dataset
        # eval_dataset=RE_dev_dataset,  # evaluation dataset
        # compute_metrics=compute_metrics,  # define metrics function
        # compute_loss = nn.CrossEntropyLoss,
        # loss_name = nn.CrossEntropyLoss,

    # )
    trainer = MyTrainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=RE_train_dataset,  # training dataset
        eval_dataset=RE_dev_dataset,  # evaluation dataset
        compute_metrics=compute_metrics,  # define metrics function
        
        loss_name='label_smoothing',            # CrossEntropy, focal, label_smoothing, f1
    )

    # train model
    trainer.train()
    model.save_pretrained("./best_model")
