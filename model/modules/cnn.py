"""
CNN modules.
"""
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """
    Depth-wise separable convolution uses less parameters
    to generate output by convolution.
    :Examples:
        >>> m = DepthwiseSeparableConv(300, 200, 5, dim=1)
        >>> input = torch.randn(32, 300, 20)
        >>> output = m(input)
    """

    def __init__(self, in_ch, out_ch, k, dim=1, bias=False):
        """
        :param in_ch: input hidden dimension size
        :param out_ch: output hidden dimension size
        :param k: kernel size
        :param dim: default 1. 1D conv or 2D conv
        :param bias: default False. Add bias or not
        """
        super().__init__()
        if dim == 1:
            self.depthwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=in_ch,
                kernel_size=k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv1d(
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, padding=0, bias=bias)
        elif dim == 2:
            self.depthwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=in_ch,
                kernel_size=k, groups=in_ch, padding=k // 2, bias=bias)
            self.pointwise_conv = nn.Conv2d(
                in_channels=in_ch, out_channels=out_ch,
                kernel_size=1, padding=0, bias=bias)
        else:
            raise Exception("Incorrect dimension!")

    def forward(self, x):
        """
        :Input: (batch_num, in_ch, seq_length)
        :Output: (batch_num, out_ch, seq_length)
        """
        return self.pointwise_conv(self.depthwise_conv(x))
