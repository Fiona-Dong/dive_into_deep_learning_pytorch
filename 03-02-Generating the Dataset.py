# %%
import numpy as np
import torch
from torch.utils import data
from torch import nn

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

# %%
def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))

# %%
net = nn.Sequential(nn.Linear(in_features=2, out_features=1))
# %%
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
# %%
loss = nn.MSELoss()
# %%
trainer = torch.optim.SGD(net.parameters(), lr=0.03)
# %%
num_epochs = 3

for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(features, labels)  
    print("epoch %d, loss %.4f" % (epoch, l))

# %%

# %%
