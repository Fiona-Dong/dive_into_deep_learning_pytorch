#%%
import torch
from torch import nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# %%
trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = datasets.ImageFolder('/Users/dongyajie/Desktop/files/data_t', transform=trans)
train_dl = DataLoader(dataset, batch_size=1, shuffle=True)
val_dl = DataLoader(dataset, batch_size=1, shuffle=False)

for batch in train_dl:
    X, y = batch
    print(X.shape)
    print(y.shape)
    break

# %%
net = models.resnet18()
net.fc = nn.Linear(in_features=512, out_features=2, bias=True)

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
loss = nn.CrossEntropyLoss()

# %%
for images, labels in train_dl:
    optimizer.zero_grad()
    outputs = net(images)
    l = loss(outputs, labels)
    l.backward()
    print(l)
    optimizer.step()

# %%

# %%
net.train()
net.eval()
# %%
datasets.ImageFolder("", trans, )