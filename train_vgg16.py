import torch as th
import torch.nn as nn
import torchvision.models as models
import numpy as np

from math import ceil
import random

import argparse
from os.path import exists
import sys

from tqdm import tqdm


def get_vgg16_modified() -> nn.Module:
    vgg16 = models.vgg16()

    vgg16.classifier[-1] = nn.Linear(4096, 1)
    vgg16.classifier.add_module("7", nn.Sigmoid())
    return vgg16


# 1000 images en 224 * 224 * 3 ~= 42Go
def main():
    # Create argument parser
    parser = argparse.ArgumentParser("Train VGG16 Main")
    parser.add_argument("-d", "--data-path", type=str, required=True, dest="data_path")
    parser.add_argument("-l", "--label-path", type=str, required=True, dest="label_path")
    parser.add_argument("-o", "--output-model-path", type=str, required=True, dest="output_model_path")

    # Parse args
    args = parser.parse_args()

    # Get args
    data_path = args.data_path
    label_path = args.label_path
    output_model_path = args.output_model_path

    # Test if numpy data and labels exist
    if not exists(data_path):
        raise FileNotFoundError("Numpy data file doesn't exist ({}) !".format(data_path))
    if not exists(label_path):
        raise FileNotFoundError("Numpy label file doesn't exist ({}) !".format(label_path))

    print("Loading VGG16 model...")
    # Create modified VGG16 model
    vgg16 = get_vgg16_modified()

    # Define loss function
    loss_fn = nn.MSELoss()

    # Pass to cuda model and loss function
    vgg16.cuda()
    loss_fn.cuda()

    # Create optimizer
    optim = th.optim.SGD(vgg16.parameters(), lr=1e-4)

    # Epoch number and batch size
    nb_epoch = 3
    batch_size = 32

    print("Loading data...")
    # Load data
    data = np.load(data_path)
    labels = np.load(label_path)

    print("Shuffle data...")
    # Shuffle data
    # https://stackoverflow.com/questions/6127503/shuffle-array-in-c
    # to prevent memory allocation
    for i in tqdm(range(data.shape[0] - 1)):
        j = i + random.randint(0, sys.maxsize) // (sys.maxsize // (data.shape[0] - i) + 1)

        data[i, :, :, :], data[j, :, :, :] = data[j, :, :, :], data[i, :, :, :]
        labels[i], labels[j] = labels[j], labels[i]

    # Compute batch number
    nb_batch = ceil(data.shape[0] / batch_size)

    print("Starting training...")
    # Train loop
    for e in range(nb_epoch):

        sum_loss = 0

        # Batch loop
        for i_b in tqdm(range(nb_batch)):
            # Get batch indexes
            i_min = i_b * batch_size
            i_max = (i_b + 1) * batch_size
            i_max = i_max if i_max < data.shape[0] else data.shape[0]

            # Slice data to get batch
            batch = data[i_min:i_max, :, :, :]
            batch = batch.transpose(0, 3, 1, 2)
            batch = th.tensor(batch).cuda().float() / 255.

            # And labels
            batch_label = th.tensor(labels[i_min:i_max]).cuda().float()

            # Forward
            out = vgg16(batch).squeeze(1)

            # Compute loss
            loss = loss_fn(out, batch_label)

            # Backward
            loss.backward()

            # Upgrade weights
            optim.step()

            sum_loss += loss.item()

        print("Epoch {}, loss = {}".format(e, sum_loss / nb_batch))

    th.save(vgg16.state_dict(), output_model_path)


if __name__ == "__main__":
    main()
