# %%
import torch
import torchvision
from torchvision import transforms
from torch.utils import data

import matplotlib.pyplot as plt
%matplotlib inline

# %%
# Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor
trans = transforms.ToTensor()

minst_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
minst_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

# %%
print(len(minst_train), len(minst_test))
print(minst_train[0][0].shape)

# %%
def get_fashion_mnist_labels(labels):
    """Return text labels for the Fashion-MNIST dataset."""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # Tensor Image
            ax.imshow(img.numpy())
        else:
            # PIL Image
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    # return axes
    plt.show()

# %%
X, y = next(iter(data.DataLoader(minst_train, batch_size=18)))
show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))

# %%
def get_dataloader_workers():
    return 4

batch_size = 256
train_iter = data.DataLoader(minst_train, batch_size=batch_size, shuffle=True, num_workers=get_dataloader_workers())

# %%
def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor()]
    
    if resize:
        trans.insert(0, transforms.Resize(resize))

    trans = transforms.Compose(trans)

    minst_train = torchvision.datasets.FashionMNIST(root="./data", train=True, transform=trans, download=True)
    minst_test = torchvision.datasets.FashionMNIST(root="./data", train=False, transform=trans, download=True)

    return (data.DataLoader(minst_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()), 
    data.DataLoader(minst_test, batch_size, shuffle=False, num_workers=get_dataloader_workers()))

# %%
train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
for X, y in train_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

# %%
