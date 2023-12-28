import math
import torch
import torch.nn as nn
from einops import repeat

try:
    import awq_ext  # with CUDA kernels
    AWQ_INSTALLED = True
except:
    AWQ_INSTALLED = False


def make_divisible(c, divisor):
    return (c + divisor - 1) // divisor

def calculate_zeros_width(in_features, group_size=128, pack_num=8):
    if group_size >= 128:
        size_multiplier = 1
    elif group_size == 64:
        size_multiplier = 2
    elif group_size == 32:
        size_multiplier = 4
    else:
        raise NotImplementedError
    
    base_width = make_divisible(in_features // group_size, pack_num)
    base_width = make_divisible(base_width, size_multiplier) * size_multiplier
    return base_width

class WQLinear_GEMM(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.pack_factor = (32 // self.w_bit)
        
        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer('qweight', torch.zeros((in_features, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((in_features // self.group_size, out_features // (32 // self.w_bit)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((in_features // self.group_size, out_features), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales
        
        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[idx // group_size]) / awq_linear.scales[idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros((zeros.shape[0], zeros.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=zeros.device)
        
        for col in range(zeros.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 2, 4, 6, 1, 3, 5, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        
        return awq_linear

    def dequantize(self):
        assert self.qweight.shape == (self.out_features, self.in_features // self.pack_factor)
        assert self.qzeros.shape == (self.out_features // self.group_size, self.in_features // self.pack_factor)
        assert self.scales.shape == (self.out_features // self.group_size, self.in_features)

        qw_unpacked = repeat(self.qweight, 'i j -> i (j p)', p=self.pack_factor)
        qzeros_unpacked = repeat(self.qzeros, 'i j -> i (j p)', p=self.pack_factor)
        # TODO: This packing order never shows up anywhere in the AWQ source code
        pack_order = [0, 4, 1, 5, 2, 6, 3, 7]
        shifter = torch.tensor(pack_order, dtype=torch.int32, device=self.qweight.device)
        shifter *= 4
        shifter = repeat(shifter, 'i -> (n i)', n=self.in_features // self.pack_factor)
        mask = (1 << 4) - 1

        qw_unpacked = (qw_unpacked >> shifter[None, :]) & mask
        qzeros_unpacked = (qzeros_unpacked >> shifter[None, :]) & mask

        qzeros_unpacked = repeat(qzeros_unpacked, 'i j -> (i g) j', g=self.group_size)
        scales = repeat(self.scales, 'i j -> (i g) j', g=self.group_size)

        dequantized_weights = scales.to(torch.float32) * (qw_unpacked.to(torch.float32) - qzeros_unpacked.to(torch.float32))

        return dequantized_weights.to(torch.float16)

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )

        input_dtype = x.dtype
        if input_dtype != torch.float16:
            x = x.half()
        
        if AWQ_INSTALLED:
            out = awq_ext.gemm_forward_cuda(x.reshape(-1, x.shape[-1]), self.qweight, self.scales, self.qzeros, 8)
        else:
            out = nn.functional.linear(x, self.dequantize(), bias=None)
        
        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )


class WQLinear_GEMV(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev):
        super().__init__()
        
        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")
        
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.split_k_iters = 8

        # quick sanity check (make sure aligment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0
        pack_num = (32 // self.w_bit)

        self.register_buffer('qweight', torch.zeros((out_features, in_features // pack_num), dtype=torch.int32, device=dev))
        self.register_buffer('qzeros', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size)), dtype=torch.int32, device=dev))
        self.register_buffer('scales', torch.zeros((out_features, calculate_zeros_width(in_features, self.group_size) * pack_num), dtype=torch.float16, device=dev))
        if bias:
            self.register_buffer('bias', torch.zeros((out_features), dtype=torch.float16, device=dev))
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(w_bit, group_size, linear.in_features, linear.out_features, linear.bias is not None, linear.weight.device)
        if init_only:  # just prepare for loading sd
            return awq_linear
        
        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None  
        scale_zeros = zeros * scales

        pack_num = 32 // awq_linear.w_bit
        qscales = torch.zeros(
            (scales.shape[0], calculate_zeros_width(linear.in_features, group_size) * pack_num),
            dtype=torch.float16,
            device=scales.device
        )
        qscales[:, :scales.shape[1]] = scales
        awq_linear.scales = qscales
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()
        
        intweight = []
        for idx in range(awq_linear.in_features):
            intweight.append(torch.round((linear.weight.data[:, idx] + scale_zeros[:, idx // group_size]) / awq_linear.scales[:, idx // group_size]).to(torch.int)[:, None])
        intweight = torch.cat(intweight, dim=1)
        intweight = intweight.to(dtype=torch.int32)
        qweight = torch.zeros((intweight.shape[0], intweight.shape[1] // 32 * awq_linear.w_bit), dtype=torch.int32, device=intweight.device)           
         
        for col in range(intweight.shape[1] // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                qweight_col = intweight[:, col * pack_num + order_map[i]]
                qweight[:, col] |= qweight_col << (i * awq_linear.w_bit)
        awq_linear.qweight = qweight

        zeros = zeros.to(dtype=torch.int32)
        qzeros = torch.zeros(
            (zeros.shape[0], calculate_zeros_width(linear.in_features, group_size)),
            dtype=torch.int32,
            device=zeros.device,
        )
        
        for col in range((zeros.shape[1] + pack_num - 1) // pack_num):
            if awq_linear.w_bit == 4:
                order_map = [0, 1, 2, 3, 4, 5, 6, 7]
            else:
                raise NotImplementedError("Only 4-bit are supported for now.")
            for i in range(pack_num):
                if col * pack_num + order_map[i] >= zeros.shape[1]:
                    continue
                qzero_col = zeros[:, col * pack_num + order_map[i]]
                qzeros[:, col] |= qzero_col << (i * awq_linear.w_bit)
        awq_linear.qzeros = qzeros
        return awq_linear

    @torch.no_grad()
    def forward(self, x):
        out_shape = x.shape[:-1] + (self.out_features, )
        inputs = x.reshape(-1, x.shape[-1])

        input_dtype = inputs.dtype
        if input_dtype != torch.float16:
            inputs = inputs.half()
        
        if inputs.shape[0] > 8:
            out = awq_ext.gemmv2_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size, self.split_k_iters)
        else:
            out = awq_ext.gemv_forward_cuda(inputs, self.qweight, self.scales, self.qzeros, self.group_size)

        if input_dtype != torch.float16:
            out = out.to(dtype=input_dtype)
        
        out = out + self.bias if self.bias is not None else out
        return out.reshape(out_shape)
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, w_bit={}, group_size={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.w_bit, self.group_size
        )

def generate_random_data(M=128, N=4096, K=4096, device="mps"):
    MAX_INT32 = 0x7fffffff
    MIN_INT32 = -MAX_INT32 - 1
    GROUP_SIZE = 128
    PACK_FACTOR = 8

    inputs = torch.randn((M, K), dtype=torch.float16, device=device)
    qweight = torch.randint(
        MIN_INT32,
        MAX_INT32, (K, N // PACK_FACTOR),
        dtype=torch.int32,
        device=device
    )
    qzeros = torch.randint(
        MIN_INT32,
        MAX_INT32, (K // GROUP_SIZE, N // PACK_FACTOR),
        dtype=torch.int32,
        device=device
    )
    scales = 0.01 * torch.randn(
        (K // GROUP_SIZE, N),
        dtype=torch.float16,
        device=device
    )

    return inputs, qweight, scales, qzeros

if __name__ == '__main__':
    inputs, qweight, scales, qzeros = generate_random_data()
    linear = WQLinear_GEMM(4, 128, 4096, 4096, None, "mps")
    linear(inputs)