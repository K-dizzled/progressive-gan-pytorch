import torch # noqa
from torch import nn # noqa
import math


class MinibatchStdev(nn.Module):
    def __init__(self, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def forward(self, input, *args, **kwargs):
        batch_size = input.size()[0]

        centered_input = input.mean(dim=0, keepdim=True)
        centered_input = centered_input.pow(2)

        stdev = centered_input.mean(dim=0, keepdim=True)
        stdev = stdev.add(1e-8)
        stdev = stdev.sqrt()

        stdev_broadcast = stdev.div(math.sqrt(batch_size))

        return input.sub(stdev_broadcast)


class PixelNormalization(nn.Module):
    def __init__(self, **kwargs):
        super(PixelNormalization, self).__init__(**kwargs)

    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class EqualizedLearningRate:
    # https://personal-record.onrender.com/post/equalized-lr/
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualizedLearningRate(name)

        # In applying equalized learning rate to a module's name parameter,
        # the module's name parameter is first renamed as f'{name}_orig', and
        # the name parameter itself is deleted. At this point, a forward propagation
        # through the module would fail, because it does not have a name attribute,
        # which pytorch looks for by default.

        weight = getattr(module, name)
        # noinspection PyProtectedMember
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))

        # Therefore, just before each forward propagation, a name attribute needs to be created.
        # This happens in the fn function, and to run this function just before a forward
        # propagation through the module, the function needs to be registered as a forward pre-hook,
        # using the module's register_forward_pre_hook method.

        module.register_forward_pre_hook(fn)

        return fn

    # Since a Python function is just a callable, here fn is really just an
    # instance of the EqualLR class, which has a __call__ method.
    def __call__(self, module, input):
        """
        In this method, the first thing to do is to retrieve the weight_orig parameter and
        scale it by some number, the details of which are in the EqualLR.compute_weight method.
        The important thing to observe is that the weight returned by compute_weight is just a
        torch.Tensor, not a nn.Parameter. This makes it simply a non-leaf variable in the
        computation graph, and back propagation will pass through it to reach the weight_
        orig parameter, the leaf variable to be updated during training.
        """
        weight = self.compute_weight(module)

        # Then, the obtained weight tensor is set as the the name attribute to the module.
        # This means that forward propagation can go through the module now, with parameter
        # values that are scaled versions of the original. Here it's important to create the
        # weight attribute with setattr. module.register_parameter(name=name, param=nn.Parameter(weight))
        # can also create the attribute, but this would lead to unintended behaviour.
        # The back propagation would stop at weight and would not reach weight_orig, so the
        # gradient w.r.t weight_orig would not be available.
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualizedLearningRate.apply(module, name)

    return module


class EqualizedConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EqualizedConv2d, self).__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualizedConvTranspose2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super(EqualizedConvTranspose2d, self).__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualizedLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(EqualizedLinear, self).__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvolutionLayer(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 kernel_size,
                 padding,
                 kernel_size2=None,
                 padding2=None,
                 pixel_norm=True):

        super(ConvolutionLayer, self).__init__()

        self.pad1 = padding
        self.pad2 = padding2 if padding2 is not None else padding

        self.kernel1 = kernel_size
        self.kernel2 = kernel_size2 if kernel_size2 is not None else kernel_size

        layers = [
            EqualizedConv2d(
                in_channel,
                out_channel,
                self.kernel1,
                padding=self.pad1
            )
        ]

        if pixel_norm:
            layers.append(PixelNormalization())

        layers.extend(
            [
                nn.LeakyReLU(0.1),
                EqualizedConv2d(out_channel, out_channel, self.kernel2, padding=self.pad2)
            ]
        )

        if pixel_norm:
            layers.append(PixelNormalization())

        layers.append(nn.LeakyReLU(0.1))

        self.conv = nn.Sequential(*layers)

    def forward(self, input):
        out = self.conv(input)

        return out


def upscale(input, scale_factor=2):
    return nn.functional.interpolate(
        input,
        scale_factor=scale_factor,
        mode='bilinear',
        align_corners=False
    )
