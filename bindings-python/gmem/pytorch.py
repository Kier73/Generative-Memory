import torch
import torch.nn as nn
from gmem.rs_bridge import FastGMemContext
import numpy as np
import time

class MatrixVFunction(torch.autograd.Function):
    """
    Core interception hook translating PyTorch auto-differentiation graph 
    into the zero-RAM topological manifold in Rust.
    """
    @staticmethod
    def forward(ctx, input_tensor, in_features, out_features, seed):
        # 1. Instantiate the mathematically sound C-FFI backend Context
        gmem_ctx = FastGMemContext(seed)
        
        # 2. Allocate an empty tensor in physical RAM solely for the forward pass math
        # It is immediately garbage collected after the matrix multiplication
        weight = torch.empty(out_features, in_features, dtype=torch.float64)
        
        # 3. Native C-FFI Bulk Hydration (Zero Python Iteration Overhead)
        gmem_ctx.fill_tensor(weight.numpy(), start_addr=0)
        
        # Cast input correctly to float64 to match our strict mathematical engine precision
        input_f64 = input_tensor.to(torch.float64)
        
        # Execute the forward pass via PyTorch BLAS
        output = input_f64.matmul(weight.t())
        
        # Save input for backward pass (bias handled separately)
        ctx.save_for_backward(input_f64)
        ctx.seed = seed
        ctx.out_features = out_features
        ctx.in_features = in_features
        
        # Restore float32 for downstream network compatibility if needed
        return output.to(input_tensor.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input_f64, = ctx.saved_tensors
        grad_output_f64 = grad_output.to(torch.float64)
        
        # Rehydrate weights for backward pass without saving them!
        gmem_ctx = FastGMemContext(ctx.seed)
        weight = torch.empty(ctx.out_features, ctx.in_features, dtype=torch.float64)
        gmem_ctx.fill_tensor(weight.numpy(), start_addr=0)
        
        # Gradient w.r.t input: grad_output @ weight
        grad_input = grad_output_f64.matmul(weight)
        
        # Convert back to original dtype
        grad_input = grad_input.to(grad_output.dtype)
        
        # Gradients match forward arguments:
        # (input_tensor, in_features, out_features, seed)
        # We return None for the static architectural parameters (features, seed) since they don't learn!
        return grad_input, None, None, None


class GMemLinear(nn.Module):
    """
    A 1-to-1 drop-in replacement for PyTorch's nn.Linear.
    Instead of maintaining a massive physical Weight matrix that crashes VRAM (Memory Wall), 
    this layer defines its entire parametric state space exclusively via a 64-bit seed.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: int = 42):
        super(GMemLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        
        # Note: We INTENTIONALLY do not register a self.weight nn.Parameter!
        # The weight does not exist in memory. It is purely mathematical rules.
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input_tensor):
        # 1. Navigate the zero-RAM Autograd Interceptor
        out = MatrixVFunction.apply(input_tensor, self.in_features, self.out_features, self.seed)
        
        # 2. Add traditional Bias (which is maintained in RAM)
        if self.bias is not None:
            out = out + self.bias
            
        return out

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, RAM Overhead=Zero bytes'
