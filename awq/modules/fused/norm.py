import torch
from torch import nn

try:
    import awq_ext
    AWQ_INSTALLED = True
except:
    AWQ_INSTALLED = False


class FasterTransformerRMSNorm(nn.Module):
    def __init__(self, weight, eps=1e-6):
        super().__init__()
        self.weight = weight
        self.variance_epsilon = eps

        if not AWQ_INSTALLED:
            raise Exception("Fused RMSNorm needs AWQ CUDA kernels installed to run.")

    def forward(self, x):
        output = torch.empty_like(x)
        awq_ext.layernorm_forward_cuda(x, self.weight, output, self.variance_epsilon)
        return output 
