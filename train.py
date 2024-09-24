import os
import json
import argparse
from pathlib import Path

import numpy as np

import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from neuralop.models import UNO

from rtmag.train.training import train, get_dataloaders

if __name__ == "__main__":


    #-----------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    args = parser.parse_args()

    with open(args.config) as config:
        info = json.load(config)
        for key, value in info.items():
            args.__dict__[key] = value

    base_path = args.base_path
    os.makedirs(base_path, exist_ok=True)

    np.save(os.path.join(args.base_path, "args.npy"), args.__dict__)

    log_dir = Path(args.base_path) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir)


    #-----------------------------------------------------------------------------------------
    if args.model["model_name"] == "UNO":
        model = UNO(
            hidden_channels = args.model["hidden_channels"],
            in_channels = args.model["in_channels"],
            out_channels = args.model["out_channels"],
            lifting_channels = args.model["lifting_channels"],
            projection_channels = args.model["projection_channels"],
            n_layers = args.model["n_layers"],

            factorization = args.model["factorization"],
            implementation = args.model["implementation"],
            rank = args.model["rank"],

            uno_n_modes = args.model["uno_n_modes"], 
            uno_out_channels = args.model["uno_out_channels"],
            uno_scalings = args.model["uno_scalings"],
        )
    elif args.model["model_name"] == "UNO_dxdydz":
        class UNO_dxdydz(torch.nn.Module):
            def __init__(self, args):
                super().__init__()
                self.uno = UNO(
                    hidden_channels = args.model["hidden_channels"],
                    in_channels = args.model["in_channels"],
                    out_channels = args.model["out_channels"],
                    lifting_channels = args.model["lifting_channels"],
                    projection_channels = args.model["projection_channels"],
                    n_layers = args.model["n_layers"],

                    factorization = args.model["factorization"],
                    implementation = args.model["implementation"],
                    rank = args.model["rank"],

                    uno_n_modes = args.model["uno_n_modes"], 
                    uno_out_channels = args.model["uno_out_channels"],
                    uno_scalings = args.model["uno_scalings"],
                )
                self.fc = torch.nn.Linear(5, 3)

            def forward(self, x):
                x = self.uno(x)
                x = self.fc(x)
                return x
        
        model = UNO_dxdydz(args)
    else:
        raise NotImplementedError

    optimizer = Adam(model.parameters(), lr=args.training['learning_rate'])

    CHECKPOINT_PATH = os.path.join(args.base_path, "last.pt")

    if os.path.exists(CHECKPOINT_PATH):   
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint['model_state_dict'])
        ck_epoch = checkpoint['epoch'] + 1
    else:
        ck_epoch = 0

    train_dataloader, val_dataloader = get_dataloaders(args)

    train(model, optimizer, train_dataloader, val_dataloader, ck_epoch, CHECKPOINT_PATH, args, writer)