import torch
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import numpy as np


class MNISTDataset:
    def __init__(self):
        self.mean = 33.318
        self.std = 78.5675

    def get_mnist_dataloaders(self, batch_size):
        mnist_train, mnist_test = self.get_mnist_dataset()
        mnist_train = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
        mnist_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
        return mnist_train, mnist_test

    def get_mnist_dataset(self):
        mnist_train = MNIST(root='./data', train=True, download=True,
                            transform=Compose([ToTensor(), Normalize(mean=(self.mean,), std=(self.std,))]))
        mnist_test = MNIST(root='./data/', train=False, download=True,
                           transform=Compose([ToTensor(), Normalize(mean=(self.mean,), std=(self.std,))]))
        return mnist_train, mnist_test


class MyFCNN:
    # fully connected
    def __init__(self, n_classes=10, input_dim=28*28):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.layers_dims = (input_dim, 64, 10) #(input_dim, 128, 64, 64, 10)
        self.num_layers = len(self.layers_dims)
        self.W = []
        self.b = []
        self._init_params()
        self.Z_no_activation = []
        self.H_with_activation = []

    def _init_params(self):
        for i in range(self.num_layers - 1):
            self.W.append(torch.rand(self.layers_dims[i], self.layers_dims[i+1]))
            self.b.append(torch.rand(1, self.layers_dims[i+1]))

    def _to_one_hot(self, y):
        y_one_hot = torch.zeros(len(y), self.n_classes)
        for i in range(len(y_one_hot)):
            y_one_hot[i][int(y[i])] = 1
        return y_one_hot

    def train(self, train_dataloader, test_dataloader=None, epochs_n=10):
        # train_dataloader, test_dataloader = self._load_and_prepare_mnist_dataset(batch_size)
        L_train = []
        L_test = []
        for epoch in range(epochs_n):
            # ~~~~~~~~~~ train
            train_y_preds = []
            train_y = self._to_one_hot(train_dataloader.dataset)
            for n_batch, (X_batch, y_batch) in enumerate(train_dataloader):
                # ~~~~~~~~~ FORWARD
                y_batch = self._to_one_hot(y_batch)
                y_pred = self.forward(X_batch)
                train_y_preds.append(y_pred)

                # ~~~~~~~~~ BACKWARD
                self.backward(y_batch, y_pred)
            train_y_preds = torch.concat(train_y_preds)
            train_loss = self.cross_enthropy(train_y, train_y_preds)
            L_train.append(train_loss)

            if test_dataloader is not None:
                # ~~~~~~~~~~ test
                test_y_preds = []
                test_y = self._to_one_hot(test_dataloader.dataset)
                # get test predictions
                for n_batch, (X_batch, y_batch) in enumerate(train_dataloader):
                    # ~~~~~~~~~ FORWARD
                    y_pred = self.forward(X_batch)
                    test_y_preds.append(y_pred)
                test_y_preds = torch.concat(test_y_preds)
                test_loss = self.cross_enthropy(test_y, test_y_preds)
                L_test.append(test_loss)

        return L_train, L_test

    def forward(self, X_batch):
        # for batch
        self.Z_no_activation = []
        H = torch.reshape(X_batch, (-1, self.input_dim,)).float()
        self.H_with_activation = [H]  # H[0] = X; H[-1] = Y_hat
        for i in range(self.num_layers - 1):
            # linear trans.
            Z = torch.matmul(H, self.W[i]) + self.b[i]
            self.Z_no_activation.append(Z)
            if i < self.num_layers - 2:
                # activation
                H = self.relu(Z)
                self.H_with_activation.append(H)
        # softmax - get Y_hat
        Y_hat = self.softmax(Z)
        self.H_with_activation.append(Y_hat)
        return Y_hat

    def backward(self, y_batch, y_pred, sgd_learning_rate=0.1):
        # for batch
        batch_size = len(y_batch)
        dL_dW = [torch.zeros_like(w) for w in self.W]
        dL_db = [torch.zeros_like(b) for b in self.b]

        # dL_dy_pred = self.cross_enthropy_grad(y_batch, y_pred)
        # last_Z_relu_grad = self.relu_grad(self.Z_no_activation[-1])
        dL_dZ = self.cross_enthropy_softmax_grad(y_batch, y_pred)   # dL_dZ2
        chained_grads = dL_dZ
        for layer_index in reversed(range(self.num_layers - 1)):
            dL_dW[layer_index] = torch.matmul(torch.t(self.H_with_activation[layer_index]), chained_grads)
            dL_db[layer_index] = torch.matmul(torch.t(chained_grads), torch.ones(batch_size))
            if layer_index == 0:
                break
            next_Z_relu_grad = self.relu_grad(self.Z_no_activation[layer_index-1])
            chained_grads = torch.matmul(chained_grads, torch.t(self.W[layer_index]))  # dL_dH
            chained_grads = chained_grads * next_Z_relu_grad    # dL_dZ1
            # chained_grads = torch.matmul(chained_grads, torch.t(next_Z_relu_grad))     # dL_dZ1

        # update params - SGD step
        self.W = [w - sgd_learning_rate * gw for w, gw in zip(self.W, dL_dW)]
        self.b = [b - sgd_learning_rate * gb for b, gb in zip(self.b, dL_db)]

    def predict(self, X):
        y_pred = self.forward(X)
        y_pred = torch.Tensor([np.argmax(y) for y in y_pred])
        return y_pred

    def cross_enthropy(self, y_batch, y_pred):
        # this is for Batch
        batch_size = len(y_batch)
        loss_per_sample = torch.zeros(batch_size)
        for i in range(batch_size):
            sample_loss = torch.zeros(self.n_classes)
            for j in range(self.n_classes):
                l = y_batch[i][j] * torch.log(y_pred[i][j]) + (1 - y_batch[i][j]) * torch.log(1 - y_pred[i][j])
                sample_loss[j] = l
            loss_per_sample[i] = -sample_loss.mean()
        return loss_per_sample.sum()

    def cross_enthropy_softmax_grad(self, y_batch, y_pred):
        return y_pred - y_batch

    def relu(self, v):
        return v * (v > 0)

    def relu_grad(self, v):
        u = torch.clone(v).detach()
        u[u >= 0] = 1
        u[u < 0] = 0
        return u

    def softmax(self, v):
        # input is batch of predictions: batch_size X n_classes
        softmax_v = torch.zeros_like(v)
        for i in range(len(v)):
            numerators = torch.Tensor([torch.exp(y) for y in v[i]])
            denominator = sum(numerators)
            softmax_v[i] = numerators / denominator
        return softmax_v


def accuracy(y, y_pred):
    acc = (y == y_pred).astype(float).mean()
    return acc


if __name__ == '__main__':
    batch_size = 64
    epoch_n = 10
    mnist = MNISTDataset()
    train_dataloader, test_dataloader = mnist.get_mnist_dataloaders(batch_size=batch_size)
    nn = MyFCNN()
    nn.train(train_dataloader, test_dataloader, epochs_n=epoch_n)

    X_train = train_dataloader.dataset.data
    y_train = train_dataloader.dataset.labels
    X_test = test_dataloader.dataset.data
    y_test = test_dataloader.dataset.labels

    y_pred_train = nn.predict(train_dataloader.dataset.data)
    y_pred_test = nn.predict(test_dataloader.dataset.data)
    train_acc = accuracy(y_train, y_pred_train)
    test_acc = accuracy(y_test, y_pred_test)
    print(f"{train_acc=}, {test_acc=}")


