import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import numpy as np
from hw1_316375872_train import MNISTDataset, MyFCNN


def Q2():
    mnist = MNISTDataset()
    train_set, test_set = mnist.get_mnist_dataset()
    n = 128
    X_train = train_set.data[:n]
    y_bernulli = torch.bernoulli(n)
    epoch_n = 100
    nn = MyFCNN()
    train_dataloader = DataLoader({"X": X_train, "y": y_bernulli}, batch_size=n, shuffle=False)
    nn.train(train_dataloader, epochs_n=epoch_n)




