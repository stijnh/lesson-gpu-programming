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

PyTorch, often imported as `torch`, is a popular open-source Python library commonly used for machine learning and AI. It provides many features for working with tensors, automatic differentiation, neural networks, and GPU acceleration. PyTorch was originally developed by Meta, and is now maintained by the PyTorch Foundation, which is part of the Linux Foundation.

You can read more about PyTorch here: https://pytorch.org/

PyTorch has many features related to AI and training neural networks. However, in this episode, we will focus on running general-purpose code on the GPU using tensors.

# Getting started

First, we need to import `torch`:

~~~python
import torch
~~~

Next, we can check whether PyTorch detected a GPU:

~~~python
print("GPU available:", torch.cuda.is_available())
~~~

After this, we can create a device and print its name. The device determines where PyTorch stores data and performs computations. In our case, we will use `cuda:0`, which refers to the NVIDIA GPU with index 0.

~~~python
# Initialize the device
device = torch.device("cuda:0")

# Print the device name
name = torch.cuda.get_device_name(device)
print("GPU detected:", name)
~~~

This shows the full name of the GPU. For example:

~~~output
GPU detected: NVIDIA A100-SXM4-40GB MIG 3g.20gb
~~~

Next, we create a tensor. A tensor is similar to a NumPy array. We can create a tensor using `torch.tensor`, and place it on the GPU using `device=device`.

~~~python
# Create a tensor on the GPU containing the numbers [1, 2, 3, 4]
x = torch.tensor([1.0, 2, 3, 4], device=device)

# Print the number of elements
print(x.numel())

# Print the data type of the elements
print(x.dtype)
~~~

This should print `4` and `torch.float32`. Notice that PyTorch’s default floating-point type is usually 32-bit, while NumPy often defaults to 64-bit floating-point numbers. This is useful on GPUs, where 32-bit operations are usually faster. You can change the default floating-point dtype using `torch.set_default_dtype(torch.float64)`, which can be useful when matching NumPy code.


# Working with tensors

To convert a NumPy array into a PyTorch tensor, we can use `torch.from_numpy`. The resulting tensor shares memory with the NumPy array.

To convert a tensor back into a NumPy array, we can use `tensor.cpu().numpy()`. Since NumPy arrays always live in CPU memory, CUDA tensors must first be copied back to the CPU using `.cpu()`.

~~~python
import numpy as np

# Create NumPy array
a = np.array([5, 6, 7, 8])

# Convert NumPy -> PyTorch tensor (CPU)
b = torch.from_numpy(a)

# Copy CPU -> GPU
c = b.to(device)

# Copy GPU -> CPU
d = c.cpu()

# Convert back to NumPy array
e = d.numpy()

# They should be equal
print(np.all(a == e))
~~~

This should print `True`, since the conversion from NumPy to PyTorch and back is lossless.

There are also many ways to create a tensor directly on the GPU, many are similar to functions in NumPy. These function create the data directly in GPU memory, without copy from the CPU. Here are some examples.

~~~python
# Gives [0.0, 0.0, 0.0, 0.0, ...]
a = torch.zeros(10, device=device)

# Gives [1.0, 1.0, 1.0, 1.0, ...]
b = torch.ones(10, device=device)

# Gives [42.0, 42.0, 42.0, 42.0, ...]
c = torch.fill(10, 42.0, device=device)

# Gives [0.3226, 0.7758, 0.6456, ...]
d = torch.rand(10, device=device)

# Gives an unitialized array
e = torch.empty(10, device=device)

# Gives [0, 1, 2, 3, 4, ...]
f = torch.arange(10, device=device)

# Gives [0.0000, 0.1111, 0.2222, 0.3333, ..., 1.0000]
g = torch.linspace(0, 1, 10, device=device)
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

Slicing is also supported, similar to NumPy. 

~~~python
x = torch.rand(5, 5, device=device)

# Print the first row
print(x[0,:])

# Print the first column
print(x[:,0])
~~~

~~~output
tensor([0.6239, 0.7711, 0.9718, 0.5485, 0.9743], device='cuda:0')
tensor([0.6239, 0.9538, 0.6850, 0.3817, 0.3251], device='cuda:0')
~~~

Notice how, after slicing, the results are again tensors. as these are view. 
Modifying the slice will also modify the original tensor.

Tensors also support many mathematical operations. Note that

~~~python
x = torch.ones(5, device=device)
y = torch.arange(5, device=device)

print("x+y:", x + y)
print("x-y:", x - y)
print("x*y:", x * y)
print("x/y:", x / y)
print("sqrt:", torch.sqrt(y))
print("exp:", torch.exp(y))
print("sin:", torch.sin(y))
~~~

~~~output
x+y: tensor([1., 2., 3., 4., 5.], device='cuda:0')
x-y: tensor([ 1.,  0., -1., -2., -3.], device='cuda:0')
x*y: tensor([0., 1., 2., 3., 4.], device='cuda:0')
x/y: tensor([   inf, 1.0000, 0.5000, 0.3333, 0.2500], device='cuda:0')
sqrt:tensor([0.0000, 1.0000, 1.4142, 1.7321, 2.0000], device='cuda:0')
exp: tensor([ 1.0000,  2.7183,  7.3891, 20.0855, 54.5982], device='cuda:0')
sin: tensor([ 0.0000,  0.8415,  0.9093,  0.1411, -0.7568], device='cuda:0')
~~~

Be aware that you cannot mix NumPy arrays and GPU tensors in these operations. 
To be able to do this, you either need to use `.cpu()` to convert your GPU tensor into a CPU tensor or, alternatively, copy your NumPy array into a GPU tensor:

~~~python
x = torch.rand(10, device=device)
y = np.arange(10)

# This is not allowed
try:
    print(x + y)
except Exception as e:
    print("error:", e)

# This returns a tensor on the CPU
print("CPU:", x.cpu() + y)

# This returns a tensor on the GPU
print("GPU:", x + torch.tensor(y, device=device))
~~~

~~~output
error: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
CPU: tensor([0.6511, 1.7587, 2.4972, 3.3509, 4.5963])
GPU: tensor([0.6511, 1.7587, 2.4972, 3.3509, 4.5963], device='cuda:0')
~~~

# Using `torch.compile`
An important feature introduced in PyTorch 2.0 is the `@torch.compile` decorator. This decorator can be applied that Python function and tells PyTorch to compile the PyTorch operations inside the function to a more optimized version before execution it. Instead of running the function line-by-line in Python, PyTorch analyzes the operations inside the function and builds a more efficient computation graph. It can then apply various optimizations, such as fusing multiple operations together, reducing memory usage, and generating more efficient GPU kernels.

This process can significantly improve performance, especially for functions that perform many tensor operations. The compiled function can therefore run much faster than the original Python version, while keeping the code easy to read and write.

Below is a simple example. We create two large tensors on the GPU and define a function that performs several mathematical operations on them. By adding the `@torch.compile` decorator, PyTorch will attempt to optimize the function automatically.

~~~python
@torch.compile
def compute(a, b):
    return (a + b) + torch.sin(a) * torch.cos(b)
    
    
a = torch.rand(1_000_000, device=device)
b = torch.rand(1_000_000, device=device)
c = compute(a, b)

print("result:", c)
~~~


[troubleshooting](https://docs.pytorch.org/docs/2.11/user_guide/torch_compiler/compile/programming_model.html)



To see what PyTorch is doing internally, we can use `set_logs` that will print what happens under the hood.

~~~python
torch._logging.set_logs(graph_code=True)
compute(a, b)
~~~

~~~output
[__kernel_code] import triton
[__kernel_code] import triton.language as tl
[__kernel_code] 
[__kernel_code] from torch._inductor.runtime import triton_helpers, triton_heuristics
[__kernel_code] from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
[__kernel_code] from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
[__kernel_code] triton_helpers.set_driver_to_gpu()
[__kernel_code] 
[__kernel_code] @triton.jit
[__kernel_code] def triton_poi_fused_add_cos_mul_sin_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
[__kernel_code]     xnumel = 1000000
[__kernel_code]     xoffset = tl.program_id(0) * XBLOCK
[__kernel_code]     xindex = xoffset + tl.arange(0, XBLOCK)[:]
[__kernel_code]     xmask = xindex < xnumel
[__kernel_code]     x0 = xindex
[__kernel_code]     tmp0 = tl.load(in_ptr0 + (x0), xmask)
[__kernel_code]     tmp1 = tl.load(in_ptr1 + (x0), xmask)
[__kernel_code]     tmp2 = tmp0 + tmp1
[__kernel_code]     tmp3 = libdevice.sin(tmp0)
[__kernel_code]     tmp4 = libdevice.cos(tmp1)
[__kernel_code]     tmp5 = tmp3 * tmp4
[__kernel_code]     tmp6 = tmp2 + tmp5
[__kernel_code]     tl.store(out_ptr0 + (x0), tmp6, xmask)
[__kernel_code]    
~~~




