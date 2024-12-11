from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # TODO: Implement for Task 4.3.
    H_ = height // kh
    W_ = width // kw

    input = input.contiguous()
    input = input.view(batch, channel, H_, kh, W_, kw)
    input = input.permute(0, 1, 2, 4, 3, 5).contiguous()
    input = input.view(batch, channel, H_, W_, kh * kw)

    return input, H_, W_


# TODO: Implement for Task 4.3.
def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Tiled average pooling 2D."""
    batch, channel, height, width = input.shape
    new_input, new_height, new_width = tile(input, kernel)
    pooled = new_input.sum(dim=4) / (kernel[0] * kernel[1])
    return pooled.view(batch, channel, new_height, new_width)


max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return a one-hot tensor of the argmax along a dimension."""
    maxima = max_reduce(input, dim)
    return maxima == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        d = int(dim[0])
        out = max_reduce(input, d)
        ctx.save_for_backward(input, dim)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        input, dim = ctx.saved_values
        d = int(dim[0])
        max_positions = argmax(input, d)
        return grad_output * max_positions, 0


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    input = input.exp()
    return input / input.sum(dim)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    tmp = input.exp().sum(dim).log()
    return input - tmp


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Max pooling using tile and max."""
    new_input, new_height, new_width = tile(input, kernel)
    return max(new_input, 4).view(input.shape[0], input.shape[1], new_height, new_width)


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    if ignore or rate is None or input.shape is None:
        return input
    else:
        return input * (rand(input.shape, backend=input.backend) > rate)
