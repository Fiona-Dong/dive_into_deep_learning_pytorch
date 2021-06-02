# %%
import torch
# %% [markdown]
### Backward for Non-Scalar Variables
# %%
# Same as `x = torch.arange(4.0, requires_grad=True)`
x = torch.arange(4.0)
x.requires_grad_(True)
# The default value is None
print(x.grad)
# %%
y = 2 * torch.dot(x, x)
y.backward()
print(x.grad)
# %%
x.grad == 4 * x
# %%
# clear the previous gradient
x.grad.zero_()
y = x * x
y.sum().backward()
print(x.grad)
# %%
# %% [markdown]
### Detaching Computation
# %%
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad)
print(x.grad == x * x)
print(x.grad == u)
# %%
# %% [markdown]
### Computing the Gradient of Python Control Flow
# %%
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
# %%
a = torch.rand(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad)
print(a.grad == d/a)
