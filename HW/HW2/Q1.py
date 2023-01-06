import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


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

    def get_datasets(self):
        train_dataset = CIFAR10(root='./data/', train=True, transform=self.transform, download=True)
        test_dataset = CIFAR10(root='./data/', train=False, transform=self.transform, download=True)
        return train_dataset, test_dataset

    def get_dataloaders(self, return_datasets=False):
        train_dataset, test_dataset = self.get_datasets()
        # Data Loader (Input Pipeline)
        train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False)
        if return_datasets:
            return train_loader, test_loader, train_dataset, test_dataset
        return train_loader, test_loader


class CNN(nn.Module):
    def __init__(self, num_epochs=5, batch_size=100, learning_rate=0.001):
        super(CNN, self).__init__()

        # Hyper Parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(5,), padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(5,), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(8 * 8 * 32, 10)
        self.dropout = nn.Dropout(p=0.5)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return self.logsoftmax(out)


def train_model_q1():
    # Hyper Parameters
    num_epochs = 5
    batch_size = 100
    learning_rate = 0.001

    # init network
    cnn = CNN()
    if torch.cuda.is_available():
        cnn = cnn.cuda()

    # get data
    cifar10 = CIFAR10Dataset(batch_size=batch_size)
    train_loader, test_loader, train_dataset, test_dataset = cifar10.get_dataloaders(return_datasets=True)

    # Loss and Optimizer
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    print('number of parameters: ', sum(param.numel() for param in cnn.parameters()))

    # train
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            # Forward + Backward + Optimize
            outputs = cnn(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                      % (epoch + 1, num_epochs, i + 1, len(train_dataset) // batch_size, loss.data))

    # evaluate
    evaluate(cnn=cnn, loader=train_loader, loader_type="Train ")
    evaluate(cnn=cnn, loader=test_loader, loader_type="Test ")


def evaluate(cnn, loader, loader_type="Test "):
    cnn.eval()
    correct = 0
    total = 0

    for images, labels in loader:
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = cnn(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print(f'{loader_type}Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))


def main():
    train_model_q1()


if __name__ == '__main__':
    main()
