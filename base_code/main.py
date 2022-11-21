import argparse
import os
import re
from datetime import datetime, timedelta

import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf

from train import *

time_ = datetime.now() + timedelta(hours=9)
time_now = time_.strftime("%m%d%H%M")

wandb_dict = {
    "gwkim_22": "f631be718175f02da4e2f651225fadb8541b3cd9",
    "rion_": "0d57da7f9222522c1a3dbb645a622000e0344d36",
    "daniel0801": "b8c4d92272716adcb1b2df6597cfba448854ff90",
    "seokhee": "c79d118b300d6cff52a644b8ae6ab0933723a59f",
    "dk100": "263b9353ecef00e35bdf063a51a82183544958cc",
}


def main():
    wandb.init()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="base_config")
    args, _ = parser.parse_known_args()

    cfg = OmegaConf.load(f"./config/{args.config}.yaml")

    wandb.login(key=wandb_dict[cfg.wandb.wandb_username])
    model_name_ch = re.sub("/", "_", cfg.model.model_name)

    # wandb Project name
    train()


if __name__ == "__main__":
    main()
