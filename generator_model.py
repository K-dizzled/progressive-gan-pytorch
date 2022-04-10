import torch # noqa
from torch import nn # noqa
from custom_layers import *


class Generator(nn.Module):
    def __init__(self,
                 input_code_dim=128,
                 in_channel=128,
                 pixel_norm=True,
                 tanh=True):

        super(Generator, self).__init__()

        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            EqualizedConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            PixelNormalization(),
            nn.LeakyReLU(0.1)
        )

        self.progression = nn.ModuleList(
            [
                ConvolutionLayer(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm),
                ConvolutionLayer(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm),
                ConvolutionLayer(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm),
                ConvolutionLayer(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm),
                ConvolutionLayer(in_channel, in_channel // 2, 3, 1, pixel_norm=pixel_norm),
                ConvolutionLayer(in_channel // 2, in_channel // 4, 3, 1, pixel_norm=pixel_norm),
                ConvolutionLayer(in_channel // 4, in_channel // 4, 3, 1, pixel_norm=pixel_norm),
            ]
        )

        self.to_rgb = nn.ModuleList(
            [
                EqualizedConv2d(in_channel, 3, 1),
                EqualizedConv2d(in_channel, 3, 1),
                EqualizedConv2d(in_channel, 3, 1),
                EqualizedConv2d(in_channel // 2, 3, 1),
                EqualizedConv2d(in_channel // 4, 3, 1),
                EqualizedConv2d(in_channel // 4, 3, 1)
            ]
        )

        self.max_step = 6

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def progress(self, feat, module):
        out = upscale(feat)
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=0, alpha=-1):
        step = min(self.max_step, step)

        outputs = [None for _ in range(self.max_step + 1)]

        outputs[0] = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        outputs[0] = self.progression[0](outputs[0])

        outputs[1] = self.progress(outputs[0], self.progression[1])
        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(outputs[1]))
            return self.to_rgb[0](outputs[1])

        current = 2
        while current <= step:
            outputs[current] = self.progress(outputs[current - 1], self.progression[current])
            current += 1

        return self.output(
            outputs[step - 1],
            outputs[step],
            self.to_rgb[step - 2],
            self.to_rgb[step - 1], alpha
        )
