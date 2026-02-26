import torch
import torch.nn as nn
import math
from .function import GMemFunction

class GMemLinear(nn.Module):
    """
    Virtualized Linear Layer Drop-in Replacement.
    
    Acts exactly like `torch.nn.Linear(in_features, out_features)`, but requires
    **Zero Bytes** of RAM/VRAM to store the weight parameters. The weights exist
    purely as a 64-bit mathematical seed, and are hydrated directly via `gmem_rs`
    during the forward network pass.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, seed: int = 0x9E3779B97F4A7C15):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
            # Standard Kaiming uniform initialization for bias
            fan_in = in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Pass the generation directly into our Autograd Hook
        return GMemFunction.apply(
            input,
            self.seed,
            self.in_features,
            self.out_features,
            self.bias
        )
        
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, seed={self.seed:#018x}, zero_ram={True}'
