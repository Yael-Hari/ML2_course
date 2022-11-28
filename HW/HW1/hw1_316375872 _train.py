import torch
from torchvision.datasets import MNIST


class MyNN:

    def __init__(self, n_classes=10, input_dim=28*28):
        self.n_classes = n_classes
        self.input_dim = input_dim
        self.layers_dims = (input_dim, 128, 64, 64, 10)
        self.num_layers = len(self.layers_dims)
        self.W = []
        self.b = []
        self._init_params()
        self.Z_outputs_before_activation = []
        self.H_outputs_after_activation = []

    def _init_params(self):
        for i in range(self.num_layers - 1):
            self.W.append(torch.rand(self.layers_dims[i+1], self.layers_dims[i]))
            self.b.append(torch.rand(self.layers_dims[i+1]))

    def _load_mnist_dataset(self):
        mnist_train = MNIST(root='./data', train=True, download=True)
        mnist_test = MNIST(root='./data/', train=False, download=True)
        return mnist_train.train_data, mnist_train.train_labels, mnist_test.test_data, mnist_test.test_labels

    def _get_prepared_mnist_dataset(self):
        X_train, y_train, X_test, y_test = self._load_mnist_dataset()
        # reshape Xs into vecs:
        X_train = torch.reshape(X_train, (-1, self.input_dim,)).float()
        X_test = torch.reshape(X_test, (-1, self.input_dim,)).float()
        # turn ys into OneHot:
        y_train = self._to_one_hot(y_train)
        y_test = self._to_one_hot(y_test)
        return X_train, y_train, X_test, y_test

    def _to_one_hot(self, y):
        y_one_hot = torch.zeros(len(y), self.n_classes)
        for i in range(len(y_one_hot)):
            y_one_hot[i][int(y[i])] = 1
        return y_one_hot

    def train(self, epochs_n=10, batch_size=1000):
        X_train, y_train, X_test, y_test = self._get_prepared_mnist_dataset()
        for epoch in range(epochs_n):
            # num_batches = int(X_train.size(0) / batch_size)
            # for batch in range(num_batches):
            #     # ========= FOR EACH BATCH
            #     batch_start_index = batch * batch_size
            #     batch_end_index = (batch + 1) * batch_size
            #     if batch_end_index > len(X_train.size(0)):
            #         batch_end_index = len(X_train.size(0))
            #         batch_size = batch_end_index - batch_start_index

            L = torch.zeros(batch_size)
            y_pred = torch.zeros(batch_size, self.n_classes)
            # X, y = X_train[batch_start_index:batch_end_index], y_train[batch_start_index:batch_end_index]
            for i, (Xi, yi) in enumerate(zip(X_train, y_train)):
                # ~~~~~~~~~ FORWARD
                y_pred[i] = self.forward(Xi)
                loss = self.cross_enthropy(yi, y_pred[i])
                L[i] = loss

                # ~~~~~~~~~ BACKWARD
                self.backward(yi, y_pred[i])

    def forward(self, Xi):
        # for single sample
        self.Z_outputs_before_activation = []
        self.H_outputs_after_activation = [Xi]  # H[0] = X; H[-1] = Y_hat
        H = Xi
        for i in range(self.num_layers - 1):
            # linear trans.
            Z = torch.matmul(self.W[i], H) + self.b[i]
            self.Z_outputs_before_activation.append(Z)
            # activation
            H = self.relu(Z)
            self.H_outputs_after_activation.append(torch.diag(H))
        return H

    def backward(self, yi, yi_pred, sgd_learning_rate=0.1):
        # for single sample
        dL_dW = [torch.zeros_like(w) for w in self.W]
        dL_db = [torch.zeros_like(b) for b in self.b]

        dL_dy_pred = self.cross_enthropy_grad(yi, yi_pred)
        last_Z_relu_grad = self.relu_grad(self.Z_outputs_before_activation[-1])
        chained_grads = torch.matmul(dL_dy_pred, last_Z_relu_grad)
        for layer_index in reversed(range(self.num_layers - 1)):
            dL_dW[layer_index] = torch.matmul(chained_grads, self.H_outputs_after_activation[layer_index+1])
            dL_db[layer_index] = torch.matmul(chained_grads, torch.ones(len(dL_db[layer_index])))
            if layer_index == 0:
                break
            next_Z_relu_grad = self.relu_grad(self.Z_outputs_before_activation[layer_index-1])
            chained_grads = torch.matmul(chained_grads, self.W[layer_index])
            chained_grads = torch.matmul(chained_grads, next_Z_relu_grad)

        # update params
        self.W = [w - sgd_learning_rate * gw for w, gw in zip(self.W, dL_dW)]
        self.b = [b - sgd_learning_rate * gb for b, gb in zip(self.b, dL_db)]

    def cross_enthropy(self, yi, yi_pred):
        # this is for single sample
        loss = torch.zeros(self.n_classes)
        for j in range(self.n_classes):
            l = yi[j] * torch.log(yi_pred[j]) + (1 - yi[j]) * torch.log(1 - yi_pred[j])
            loss[j] = l
        return -loss.mean()

    def cross_enthropy_grad(self, yi, yi_pred):
        # this is for single sample
        dL_dyi_pred = torch.zeros(self.n_classes)
        for j in range(self.n_classes):
            dL_dyi_pred[j] = (1 / self.n_classes) * (((1 - yi[j]) / (1 - yi_pred[j])) - (yi[j] / yi_pred[j]))
        return torch.diag(dL_dyi_pred)

    def relu(self, v):
        return v * (v > 0)

    def relu_grad(self, v):
        u = torch.Tensor([_ for _ in v])
        u[u >= 0] = 1
        u[u < 0] = 0
        return torch.diag(u)

    def max_pool(self, v):
        pass

    def softmax(self):
        pass

    def cnn_layer(self, x):
        pass


if __name__ == '__main__':
    nn = MyNN()
    nn.train()
