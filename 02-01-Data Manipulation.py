# %%
import torch
# %% [markdown]
## Getting started
# %%
x = torch.arange(12)
print(x.shape)
print(x.numel())
# %%
X = x.reshape(3, 4)
print(X.shape)
# %%
x = torch.zeros(2, 3, 4)
print(x.shape)
x
# %%
x = torch.ones(2, 3, 4)
print(x.shape)
x
# %%
x = torch.rand(3, 4)
x
# %%
x = torch.tensor([[0, 1, 2], [3, 4, 5]])
print(x.shape)
x
# %% [markdown]
## Operations
# %%
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
# %%
torch.exp(x)
# %%
X==Y
# %%
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print()
print(torch.cat((X, Y), dim=1))
# %%
print(X.sum())
print(X.sum(axis=1))
print(X.sum(axis=1, keepdims=True))
# %%
x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4)
x, y, torch.dot(x, y)
# %%
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
x = torch.ones(4)
A, x, torch.mv(A, x)
# %%
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B= torch.ones(4, 3)
torch.mm(A, B)

# %% [markdown]
## Broadcasting Mechanism
# %%
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
a, b
# %%
a+b
# %% [markdown]
## Indexing and Slicing
# %%
X, X[-1], X[1:3]

# %%
X[0:2, :] = 12
X
# %% [markdown]
## Saving Memory

# %%
before = id(Y)
Y = Y + X
after = id(Y)
before==after
# %%
Z = torch.zeros_like(Y)
print(id(Z))
Z[:] = Y + X
print(id(Z))
# %%
before = id(Y)
Y += X
after = id(Y)
before==after
# %% [markdown]
## Conversion to Other Python Objects

# %%
A = X.numpy()
B = torch.tensor(A)
type(A), type(B)
# %%
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
