# %%
import torch

import matplotlib.pyplot as plt
%matplotlib inline

import random
# %% [markdown]
### Generating the Dataset
# %%
def synthetic_data(w, b, num_examples):
    """Generate y = Xw + b + noise."""
    X = torch.normal(mean=0, std=0.1, size=(num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(mean=0, std=0.01, size=y.shape)
    return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print("features: ", features.shape)
print("labels: ", labels.shape)
print()
print(features[0])
print(labels[0])
# %%
fig, ax = plt.subplots()
plt.scatter(features[:, 1], labels, s=9)
plt.show()
# %% [markdown]
### Reading the Dataset
# %%
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i+batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

# %%
batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print("X: ", X)
    print("y: ", y)
    break

# %% [markdown]
### Initializing Model Parameters
# %%
w = torch.normal(mean=0, std=0.1, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# %% [markdown]
### Defining the Model
# %%
def linreg(X, w, b):
    """The linear regression model."""
    return torch.matmul(X, w)+b
# %% [markdown]
### Defining the Loss Function
# %%
def squared_loss(y_hat, y):
    """Squared loss."""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
# %% [markdown]
### Defining the Optimization Algorithm
# %%
def sgd(params, lr, batch_size):
    # disabled gradient calculation
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            # clears old gradient from the last step
            param.grad.zero_()
# %% [markdown]
### Training
# %%
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print("epoch %d, loss %.4f" % (epoch, train_l.mean()))

# %%

# %%
