import argparse
import json
import logging
import os
import numpy as np

import torchvision
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)06d: %(levelname).1s %(pathname)s:%(lineno)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def add_parser_arguments(parser):
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs (default: 10) to run",
    )
    parser.add_argument(
        "--steps",
        default=70,
        type=int,
        metavar="N",
        help="number of steps (default: 70) per epochs to run",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256) per worker",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.001,
        type=float,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Run model in mixed precision mode.",
    )
    parser.add_argument(
        "--collective-communication",
        type=str,
        default="auto",
        choices=["auto", "nccl", "mpi", "gloo"],
        help="collective communication strategy for workers.",
    )


def mnist_dataset():
    transforms = torch.nn.Sequential(
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ConvertImageDtype(torch.float),
        torchvision.transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        ),
    )
    scripted_transforms = torch.jit.script(transforms)

    train_dataset = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            root="data",
            train=True,
            transform=scripted_transforms,
            download=True,
        )
    )

    test_dataset = torchvision.datasets.MNIST(
        root="data",
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True,
    )

    return train_dataset, test_dataset


def build_and_compile_cnn_model():
    model = torchvision.models.efficientnet_v2_s(pretrained=False)
    model.to('cuda')
    model = torch.nn.parallel.DistributedDataParallel(model)
    return model

def main():
    for epoch in range(args.start_epoch, args.epochs):
