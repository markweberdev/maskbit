"""This file contains the definition of the discriminator."""
import functools
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from modeling.modules.autoencoder import Conv2dSame


class BlurBlock(torch.nn.Module):
    def __init__(
        self,
        kernel: Tuple[int] = (1, 3, 3, 1)
    ):
        """ Initializes the blur block.

        Args:
            kernel -> Tuple[int]: The kernel size.
        """
        super().__init__()

        self.kernel_size = len(kernel)

        kernel = torch.tensor(kernel, dtype=torch.float32, requires_grad=False)
        kernel = kernel[None, :] * kernel[:, None]
        kernel /= kernel.sum()
        kernel = kernel.unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel", kernel)

    def calc_same_pad(self, i: int, k: int, s: int) -> int:
        """ Calculates the same padding for the BlurBlock.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride.

        Returns:
            pad -> int: The padding.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        
        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            out -> torch.Tensor: The output tensor.
        """
        ic, ih, iw = x.size()[-3:]
        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size, s=2)
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size, s=2)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])

        weight = self.kernel.expand(ic, -1, -1, -1)

        out = F.conv2d(input=x, weight=weight, stride=2, groups=x.shape[1])
        return out


class NLayerDiscriminatorv2(torch.nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        num_stages: int = 3,
        activation_fn: str = "leaky_relu",
        blur_resample: bool = False,
        blur_kernel_size: int = 4
    ):
        """ Initializes the NLayerDiscriminatorv2.

        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
            activation_fn -> str: The activation function.
            blur_resample -> bool: Whether to use blur resampling.
            blur_kernel_size -> int: The blur kernel size.
        """
        super().__init__()
        assert num_stages > 0, "Discriminator cannot have 0 stages"
        assert (not blur_resample) or (blur_kernel_size >= 3 and blur_kernel_size <= 5), "Blur kernel size must be in [3,5] when sampling]"

        in_channel_mult = (1,) + tuple(map(lambda t: 2**t, range(num_stages)))
        init_kernel_size = 5
        if activation_fn == "leaky_relu":
            activation = functools.partial(torch.nn.LeakyReLU, negative_slope=0.1)
        else:
            activation =  torch.nn.SiLU

        self.block_in = torch.nn.Sequential(
            Conv2dSame(
                num_channels,
                hidden_channels,
                kernel_size=init_kernel_size
            ),
            activation(),
        )

        BLUR_KERNEL_MAP = {
            3: (1,2,1),
            4: (1,3,3,1),
            5: (1,4,6,4,1),
        }

        discriminator_blocks = []
        for i_level in range(num_stages):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]
            block = torch.nn.Sequential(
                Conv2dSame(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                ),
                torch.nn.AvgPool2d(kernel_size=2, stride=2) if not blur_resample else BlurBlock(BLUR_KERNEL_MAP[blur_kernel_size]),
                torch.nn.GroupNorm(32, out_channels),
                activation(),
            )
            discriminator_blocks.append(block)

        self.blocks = torch.nn.ModuleList(discriminator_blocks)

        self.pool = torch.nn.AdaptiveMaxPool2d((16, 16))

        self.to_logits = torch.nn.Sequential(
            Conv2dSame(out_channels, out_channels, 1),
            activation(),
            Conv2dSame(out_channels, 1, kernel_size=5)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            output -> torch.Tensor: The output tensor.
        """
        hidden_states = self.block_in(x)
        for block in self.blocks:
            hidden_states = block(hidden_states)

        hidden_states = self.pool(hidden_states)

        return self.to_logits(hidden_states)


class OriginalNLayerDiscriminator(torch.nn.Module):
    """Defines a PatchGAN discriminator like in Pix2Pix as used by Taming VQGAN
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(
        self,
        num_channels: int = 3,
        hidden_channels: int = 64,
        num_stages: int = 3,
    ):
        """Initializes a PatchGAN discriminator.
        
        Args:
            num_channels -> int: The number of input channels.
            hidden_channels -> int: The number of hidden channels.
            num_stages -> int: The number of stages.
        """
        super(OriginalNLayerDiscriminator, self).__init__()
        norm_layer = torch.nn.BatchNorm2d

        sequence = [torch.nn.Conv2d(num_channels, hidden_channels, kernel_size=4, stride=2, padding=1), torch.nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_stages):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                torch.nn.Conv2d(hidden_channels * nf_mult_prev, hidden_channels * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(hidden_channels * nf_mult),
                torch.nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_stages, 8)
        sequence += [
            torch.nn.Conv2d(hidden_channels * nf_mult_prev, hidden_channels * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(hidden_channels * nf_mult),
            torch.nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            torch.nn.Conv2d(hidden_channels * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map
        self.main = torch.nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. 
        
        Args:
            x -> torch.Tensor: The input tensor.
        
        Returns:
            output -> torch.Tensor: The output tensor.
        """
        return self.main(x)




if __name__ == "__main__":
    patch_discriminator_v2 = NLayerDiscriminatorv2(num_channels=3, hidden_channels=128, num_stages=3)
    patch_discriminator_v2_blur = NLayerDiscriminatorv2(num_channels=3, hidden_channels=128, num_stages=3, blur_resample=True)
    original_discriminiator = OriginalNLayerDiscriminator(num_channels=3, hidden_channels=128, num_stages=3)
 
    from torchinfo import summary

    print("Original Discriminator")
    summary(original_discriminiator, input_size=(1,3,256,256), depth=3, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    print("Patch Discriminator v2")
    summary(patch_discriminator_v2, input_size=(1,3,256,256), depth=3, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))
    print("Patch Discriminator v2 (blur)")
    summary(patch_discriminator_v2_blur, input_size=(1,3,256,256), depth=3, col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"))


    x = torch.randn((1, 3, 256, 256)).to(next(original_discriminiator.parameters()))

    out_original = original_discriminiator(x)
    out_patch_v2 = patch_discriminator_v2(x)
    out_patch_v2_blur = patch_discriminator_v2_blur(x)

    print(f"Input shape: {x.shape}")
    print(f"Patch Discriminator v2 output shape: {out_patch_v2.shape}")
    print(f"Patch Discriminator v2 (blur) output shape: {out_patch_v2_blur.shape}")
    print(f"Original Discriminator output shape: {out_original.shape}")