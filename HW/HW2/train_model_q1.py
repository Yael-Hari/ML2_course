import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
import numpy as np


class CIFAR10Dataset:
    def __init__(self, batch_size=100):
        # Hyper Parameters
        self.batch_size = batch_size

        # Image Preprocessing
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615)),
            ]
        )

    def get_datasets(self, aug_train=False):
        train_dataset = CIFAR10(root='./data/', train=True, transform=self.transform, download=True)
        if aug_train:
            aug_train_dataset = self._get_augmented_train_dataset()
            train_dataset = ConcatDataset([train_dataset, aug_train_dataset])

        test_dataset = CIFAR10(root='./data/', train=False, transform=self.transform, download=True)
        return train_dataset, test_dataset

    def get_dataloaders(self, return_datasets=False, aug_train=False):
        train_dataset, test_dataset = self.get_datasets(aug_train=aug_train)
        # Data Loader (Input Pipeline)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        if return_datasets:
            return train_loader, test_loader, train_dataset, test_dataset
        return train_loader, test_loader

    def _get_augmented_train_dataset(self):
        aug_transform1 = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615))
            ]
        )
        aug_transform2 = transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomCrop(32, padding=6),
                transforms.RandomRotation(10),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Set the color params
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.2434, 0.2615))
            ]
        )
        aug_transform3 = transforms.Compose(
            [
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),
                transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
                transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.25),  # Set the color params
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))
            ]
        )
        aug_train_dataset1 = CIFAR10(root='./data/', train=True, transform=aug_transform1, download=True)
        aug_train_dataset2 = CIFAR10(root='./data/', train=True, transform=aug_transform2, download=True)
        aug_train_dataset3 = CIFAR10(root='./data/', train=True, transform=aug_transform3, download=True)
        aug_train_dataset = ConcatDataset([aug_train_dataset1, aug_train_dataset2, aug_train_dataset3])
        return aug_train_dataset


class CNN(nn.Module):
    def __init__(self, batch_size=100, dropoout=0.25, fc_hidden=128):
        super(CNN, self).__init__()

        # Hyper Parameters
        self.batch_size = batch_size
        self.dropoout = dropoout
        self.fc_hidden = fc_hidden

        # layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=60, kernel_size=3, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.dropout = nn.Dropout(p=self.dropoout)
        self.fc = nn.Linear(8 * 8 - 4, 10)

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


def train_model_q1():
    torch.manual_seed(42)

    # Hyper Parameters
    batch_size = 128
    # learning_rate = 0.001
    weight_decay = 0.0001
    # dropout = 0.25
    loss_func = nn.NLLLoss()

    # get data
    cifar10 = CIFAR10Dataset(batch_size=batch_size)
    train_loader, test_loader, train_dataset, test_dataset = cifar10.get_dataloaders(return_datasets=True,
                                                                                     aug_train=True)

    learning_rate = 0.001
    dropout = 0.4

    # for learning_rate, dropout in itertools.product(learning_rate_l, dropout_l):
    print(f"==========={learning_rate=}, {dropout=}===========")

    # init network
    cnn = CNN(batch_size=batch_size, dropoout=dropout)
    if torch.cuda.is_available():
        print(f"{torch.cuda.is_available()=}")
        cnn = cnn.cuda()
    else:
        print('cpu')

    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=weight_decay)

    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))

    # train
    L_train = []
    L_test = []
    Acc_train = []
    Acc_test = []
    num_epochs = 0
    epoch_test_accuracy = 0.0

    # for epoch in range(num_epochs):
    while epoch_test_accuracy <= 80.0:
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = cnn(images)
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f | '
                      % (num_epochs + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data))

        # eval epoch
        epoch_train_accuracy, epoch_train_loss = \
            evaluate(cnn=cnn, loader=train_loader, loader_type="\tTrain ", loss_func=loss_func)
        epoch_test_accuracy, epoch_test_loss = \
            evaluate(cnn=cnn, loader=test_loader, loader_type="\tTest ", loss_func=loss_func)

        Acc_train.append(epoch_train_accuracy)
        L_train.append(epoch_train_loss)
        Acc_test.append(epoch_test_accuracy)
        L_test.append(epoch_test_loss)
        num_epochs += 1

    return cnn, L_train, L_test, Acc_train, Acc_test, num_epochs


def evaluate(cnn, loader, loss_func, loader_type="Test "):
    cnn.eval()
    correct = 0.0
    total = 0.0
    loss_list = []

    for images, labels in loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().clone().detach().cpu()

        batch_loss = loss_func(outputs.clone().detach().cpu(), labels.clone().detach().cpu())
        loss_list.append(batch_loss)

    avg_loss = sum(loss_list) / len(loss_list)
    accuracy = 100 * correct / total
    print(f'{loader_type}Accuracy:', accuracy, f'{loader_type}Loss:', avg_loss)

    return accuracy, avg_loss


def plot_acc_and_loss(num_epochs, L_train, L_test, Acc_train, Acc_test):
    x = np.arange(num_epochs)
    # plot
    plt.plot(x, L_train, label="train loss")
    plt.plot(x, L_test, label="test loss")
    plt.title("Loss over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    plt.plot(x, Acc_train, label="train accuracy")
    plt.plot(x, Acc_test, label="test accuracy")
    plt.title("Accuracy over epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()


def main():
    trained_cnn, L_train, L_test, Acc_train, Acc_test, num_epochs = train_model_q1()
    # save trained model
    path = 'model_q1.pkl'
    torch.save(trained_cnn.state_dict(), path)
    plot_acc_and_loss(num_epochs, L_train, L_test, Acc_train, Acc_test)


if __name__ == '__main__':
    main()
