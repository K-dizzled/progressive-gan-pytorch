import torch # noqa
from torch import nn # noqa
from torch.nn import functional as F # noqa
from custom_layers import *


class Discriminator(nn.Module):
    def __init__(self, feat_dim=128, allow_std_dev=False):
        super(Discriminator, self).__init__()

        self.progression = nn.ModuleList(
            [
                ConvolutionLayer(feat_dim // 4, feat_dim // 4, 3, 1),
                ConvolutionLayer(feat_dim // 4, feat_dim // 2, 3, 1),
                ConvolutionLayer(feat_dim // 2, feat_dim, 3, 1),
                ConvolutionLayer(feat_dim, feat_dim, 3, 1),
                ConvolutionLayer(feat_dim, feat_dim, 3, 1),
                ConvolutionLayer(feat_dim, feat_dim, 3, 1),
                ConvolutionLayer(feat_dim + 1, feat_dim, 3, 1, 4, 0)
            ]
        )

        self.from_rgb = nn.ModuleList(
            [
                EqualizedConv2d(3, feat_dim // 4, 1),
                EqualizedConv2d(3, feat_dim // 4, 1),
                EqualizedConv2d(3, feat_dim // 2, 1),
                EqualizedConv2d(3, feat_dim, 1),
                EqualizedConv2d(3, feat_dim, 1),
                EqualizedConv2d(3, feat_dim, 1),
                EqualizedConv2d(3, feat_dim, 1)
            ]
        )

        self.n_layer = len(self.progression)
        self.minibatch_std_dev = MinibatchStdev() if allow_std_dev else None

        self.linear = EqualizedLinear(feat_dim, 1)

    @staticmethod
    def weighted_sum(input1, input2, alpha):
        out = alpha * input1 + (1 - alpha) * input2
        return out

    def forward(self, input, step=0, alpha=-1):
        out = self.from_rgb[self.n_layer - step - 1](input)

        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                out = upscale(out, scale_factor=0.5)

                if i == step and 0 <= alpha < 1:
                    skip_rgb = upscale(input, scale_factor=0.5)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = self.weighted_sum(out, skip_rgb, alpha)

        if self.minibatch_std_dev is not None:
            out = self.minibatch_std_dev(out) # noqa

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out) # noqa

        return out
