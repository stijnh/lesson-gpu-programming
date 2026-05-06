---
title: "Using PyTorch for GPU computing"
teaching: 60
exercises: 10
---

:::::: questions
- "TODO"
::::::

:::::: objectives
- "TODO"
::::::

# Introduction to PyTorch

PyTorch (or just _torch_) is a popular open-source library for Python that is commonly used for machine learning and AI tasks. It provides many features and functions and is widely used by many people. It was originally developed by Meta Inc., but is currently maintained by the PyTorch Foundation, which is a subsidiary of the Linux Foundation.

You can read more about torch here: https://pytorch.org/

PyTorch has many features related to AI and training of neural networks.
However, for this episode, we will focus on running general-prupose code on the GPU using *Tensor*s.

# Getting started
First, we need to import `torch`:


~~~python
import torch
~~~

Next, we can check if PyTorch was able to detect our GPU:

~~~python
print("GPU available:", torch.cuda.is_available())
~~~

After this, we can create a _device_ and print its name. The device determines where PyTorch stores the data and performs computations. In our case, we will use device `cuda:0`, which refers to the NVIDA GPU with index 0 that is available in your system.

~~~python
# Initialize the device
device = torch.device("cuda:0")

# Print the name
name = torch.cuda.get_device_name(device)
print("GPU detected:", name)
~~~

This shows the full name of our GPU. For example:

~~~output
GPU detected: NVIDIA A100-SXM4-40GB MIG 3g.20gb
~~~

Next, we create a tensor. A tensor is similar to a NumPy array. We can create a tensor using `torch.tensor`, where we need to specify that we want the data on the GPU using `device=device`.

~~~python
# Create tensor on the GPU containing the numbers [1,2,3,4]
x = torch.tensor([1.0, 2, 3, 4], device=device)

# Print the number of elements
print(x.numel())

# Print the data type of the elements
print(x.dtype)
~~~

Which should print `4` and `torch.float32`. Notice that the default data type (32-bit floating-point numbers) is different from NumPy's default data type (which uses 64-bit), since GPUs are much faster on 32-bit numbers. The default can be changed using `torch.set_default_dtype(torch.float64)` for compability with numpy.


# Working with tensors
To convert a NumPy array into a PyTorch tensor, we can use the function `torch.tensor`. To convert it back to a NumPy array, we can use the function `tensor.cpu().numpy()`.

~~~python
import numpy as np

# Create numpy array
a = np.array([5, 6, 7, 8])

# Convert numpy -> torch
b = torch.from_numpy(a)

# Copy CPU -> GPU
c = b.to(device)

# Copy GPU -> CPU
d = c.cpu()

# Convert back to numpy array
e = d.numpy()

# They should be equal
print(np.all(a == e))
~~~

This should print `True`, since the conversion of NumPy to PyTorch and back is lossless.A

There are also many ways to create a tensor directly on the GPU, many are similar to functions in NumPy. These function create the data directly in GPU memory, without copy from the CPU. Here are some examples.

~~~python
# Gives [0.0, 0.0, 0.0, 0.0, ...]
a = torch.zeros(10, device=device)

# Gives [1.0, 1.0, 1.0, 1.0, ...]
b = torch.ones(10, device=device)

# Gives [0.3226, 0.7758, 0.6456, ...]
c = torch.rand(10, device=device)

# Gives an unitialized array
d = torch.empty(10, device=device)

# Gives [0, 1, 2, 3, 4, ...]
e = torch.arange(10, device=device)

# Gives [0.0000, 0.1111, 0.2222, 0.3333, ..., 1.0000]
f = torch.linspace(0, 1, 10, device=device)
~~~

Tensors can be multi-dimensional. They can be 1-dimensional (vector), 2-dimensional (matrix), 3-dimensional, ...

~~~python
# This is a vector of lenght 3
x = torch.rand(3, device=device)

# This is a matrix of size 3x3
y = torch.rand((3, 3), device=device)

# This is a cube of size 3x3x3
z = torch.rand((3, 3, 3), device=device)
~~~

Slicing is also supported, similar to NumPy:

~~~python
x = torch.rand(5, 5, device=device)

# This is our matrix
print(x)

# Print the first row
print(x[0,:])

# Print the first column
print(x[:,0])
~~~

# Using `torch.compile`
From version 2.0, PyTorch supports the `@torch.compile` decorator. 

[troubleshooting](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/compile/programming_model.html)


