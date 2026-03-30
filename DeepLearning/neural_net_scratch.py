import torch

torch.manual_seed(42)


def make_dataset(n_samples=200):
    n = n_samples // 2

    class0 = torch.randn(n, 2) * 0.5 + torch.tensor([-2.0, 0.0])
    class1 = torch.randn(n, 2) * 0.5 + torch.tensor([2.0, 0.0])

    X = torch.cat([class0, class1], dim=0)
    y = torch.cat([
        torch.zeros(n, 1),
        torch.ones(n, 1)
    ], dim=0)

    return X, y


X, y = make_dataset()


def sigmoid(z):
    return 1 / (1 + torch.exp(-z))


def sigmoid_backward(sigmoid_output):
    return sigmoid_output * (1 - sigmoid_output)


def binary_cross_entropy(y_pred, y_true):
    eps = 1e-8
    loss = -(y_true * torch.log(y_pred + eps) +
             (1 - y_true) * torch.log(1 - y_pred + eps))
    return loss.mean()


def bce_backward(y_pred, y_true):
    eps = 1e-8
    return (y_pred - y_true) / ((y_pred + eps) * (1 - y_pred + eps))


class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = torch.randn(input_dim, hidden_dim) * 0.1
        self.b1 = torch.zeros(1, hidden_dim)

        self.W2 = torch.randn(hidden_dim, output_dim) * 0.1
        self.b2 = torch.zeros(1, output_dim)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_hat = sigmoid(self.z2)

        return self.y_hat

    def backward(self, X, y):
        m = X.shape[0]

        # dL/dy_hat
        dL_dyhat = bce_backward(self.y_hat, y)

        # output layer
        dyhat_dz2 = sigmoid_backward(self.y_hat)
        dL_dz2 = dL_dyhat * dyhat_dz2

        self.dW2 = self.a1.T @ dL_dz2 / m
        self.db2 = dL_dz2.mean(dim=0, keepdim=True)

        # hidden layer
        dL_da1 = dL_dz2 @ self.W2.T
        da1_dz1 = sigmoid_backward(self.a1)
        dL_dz1 = dL_da1 * da1_dz1

        self.dW1 = X.T @ dL_dz1 / m
        self.db1 = dL_dz1.mean(dim=0, keepdim=True)

    def step(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2


def train(model, X, y, epochs=1000, lr=0.1):
    for epoch in range(epochs):

        y_pred = model.forward(X)
        loss = binary_cross_entropy(y_pred, y)

        model.backward(X, y)
        model.step(lr)

        if epoch % 100 == 0:
            preds = (y_pred > 0.5).float()
            acc = (preds == y).float().mean()
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc:.4f}")


model = SimpleNN(input_dim=2, hidden_dim=8, output_dim=1)
train(model, X, y, epochs=1000, lr=0.1)