"""
Vector Addition with Triton
==========================

This example demonstrates how to implement a simple vector addition operation using Triton,
a language and compiler for writing highly efficient GPU kernels.
"""

import torch
import triton
import triton.language as tl

###############################################################################
# Defining the Triton Kernel
# --------------------------
#
# First, we define our vector addition kernel using the `@triton.jit` decorator.
# This kernel will run on the GPU and perform element-wise addition.

@triton.jit
def vector_addition_kernel(
    # Pointers to input and output tensors
    x_ptr,
    y_ptr,
    output_ptr,
    
    # Total count of elements
    n_elements,
    
    # Strides for the tensors
    x_stride: tl.constexpr,
    y_stride: tl.constexpr,
    output_stride: tl.constexpr,
    
    # Block size for parallelization
    BLOCK_SIZE: tl.constexpr,
):
    # Get unique program ID for this threadblock
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this threadblock
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate memory offsets for each tensor
    x_offsets = offsets * x_stride
    y_offsets = offsets * y_stride
    
    # Create mask for valid elements
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + x_offsets, mask=mask)
    y = tl.load(y_ptr + y_offsets, mask=mask)
    
    # Perform addition
    output = x + y
    
    # Store the result
    tl.store(output_ptr + x_offsets, output, mask=mask)

###############################################################################
# Creating the PyTorch Interface
# -----------------------------
#
# Now we create a wrapper function that interfaces with PyTorch and launches
# our Triton kernel with the appropriate parameters.

def vector_addition(x, y):
    # Verify inputs are on GPU
    assert x.is_cuda and y.is_cuda, "Input tensors must be on GPU!"
    assert x.numel() == y.numel(), "Input tensors must be the same size!"

    # Prepare output tensor
    output = torch.empty_like(x)
    
    # Define grid size based on input size
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    vector_addition_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        output_ptr=output,
        n_elements=x.numel(),
        x_stride=1,
        y_stride=1,
        output_stride=1,
        BLOCK_SIZE=128,
    )
    
    return output

###############################################################################
# Testing the Implementation
# -------------------------
#
# Let's test our implementation with both power-of-2 and non-power-of-2 sized
# tensors to verify correctness.

if __name__ == "__main__":
    # Test 1: Power of 2 size (1024)
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    
    output = vector_addition(x, y)
    
    # Verify against PyTorch implementation
    output_ref = x + y
    assert torch.allclose(output, output_ref)
    print("Success with power of 2 size (1024)!")
    print(f"{output=}")
    
    # Test 2: Non-power of 2 size (257)
    x = torch.randn(257, device='cuda')
    y = torch.randn(257, device='cuda')
    
    output_np2 = vector_addition(x, y)
    
    # Verify against PyTorch implementation
    output_ref_np2 = x + y
    assert torch.allclose(output_np2, output_ref_np2)
    print("Success with non power of 2 size (num_elems = 257)!")
    print(f"{output_np2[0:5]=}")
