"""This file contains the definition of the the autoencoder parts"""

import math
import torch
import torch.nn.functional as F

class Conv2dSame(torch.nn.Conv2d):
    """ Convolution wrapper for 2D convolutions using `SAME` padding."""
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        """ Calculate padding such that the output has the same height/width when stride=1.

        Args:
            i -> int: Input size.
            k -> int: Kernel size.
            s -> int: Stride size.
            d -> int: Dilation rate.
        """
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the convolution applying explicit `same` padding.

        Args:
            x -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        ih, iw = x.size()[-2:]

        pad_h = self.calc_same_pad(i=ih, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=iw, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return super().forward(x)


def GroupNorm(in_channels):
    """ GroupNorm with 32 groups."""
    if in_channels % 32 != 0:
        raise ValueError(f"GroupNorm requires in_channels to be divisible by 32, got {in_channels}.")
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class ResidualBlock(torch.nn.Module):
    """ Residual block with two convolutional layers."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        norm_func = GroupNorm
    ):
        """ Initializes the residual block.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels. Default is in_channels.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = self.in_channels if out_channels is None else out_channels

        self.norm1 = norm_func(self.in_channels)
        self.conv1 = Conv2dSame(self.in_channels, self.out_channels, kernel_size=3, bias=False)

        self.norm2 = norm_func(self.out_channels)
        self.conv2 = Conv2dSame(self.out_channels, self.out_channels, kernel_size=3, bias=False)

        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv2dSame(self.out_channels, self.out_channels, kernel_size=1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the residual block.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            residual = self.nin_shortcut(hidden_states)

        return hidden_states + residual


class ResidualStage(torch.nn.Module):
    """ Residual stage with multiple residual blocks."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        norm_func = GroupNorm
    ):
        """ Initializes the residual stage.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels.
            num_res_blocks -> int: Number of residual blocks.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, norm_func=norm_func))
            in_channels = out_channels

    def forward(self, hidden_states: torch.Tensor, *unused_args) -> torch.Tensor:
        """ Forward pass of the residual stage.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states)

        return hidden_states


class DownsamplingStage(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        sample_with_conv: bool = False,
        norm_func = GroupNorm
    ):
        """ Initializes the downsampling stage.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels.
            num_res_blocks -> int: Number of residual blocks.
            sample_with_conv -> bool: Whether to sample with a convolution or with a stride. Default is False.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, norm_func))
            in_channels = out_channels

        self.sample_with_conv = sample_with_conv
        if self.sample_with_conv:
            self.down_conv = Conv2dSame(in_channels, in_channels, kernel_size=3, stride=2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the downsampling stage.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states)

        if self.sample_with_conv:
            hidden_states = self.down_conv(hidden_states)
        else:
            hidden_states = F.avg_pool2d(hidden_states, kernel_size=2, stride=2)

        return hidden_states


class UpsamplingStage(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int,
        norm_func = GroupNorm
    ):
        """ Initializes the upsampling stage.

        Args:
            in_channels -> int: Number of input channels.
            out_channels -> int: Number of output channels.
            num_res_blocks -> int: Number of residual blocks.
            norm_func -> Callable: Normalization function. Default is GroupNorm.
        """
        super().__init__()

        self.res_blocks = torch.nn.ModuleList()
        for _ in range(num_res_blocks):
            self.res_blocks.append(ResidualBlock(in_channels, out_channels, norm_func))
            in_channels = out_channels

        self.upsample_conv = Conv2dSame(out_channels, out_channels, kernel_size=3)

    def forward(self, hidden_states: torch.Tensor, *unused_args) -> torch.Tensor:
        """ Forward pass of the upsampling stage.

        Args:
            hidden_states -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        for res_block in self.res_blocks:
            hidden_states = res_block(hidden_states)

        hidden_states = F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        hidden_states = self.upsample_conv(hidden_states)

        return hidden_states


class ConvEncoder(torch.nn.Module):
    def __init__(self, config):
        """ Initializes the convolutional encoder.

        Args:
            config: Configuration of the model architecture.
        """
        super().__init__()
        self.config = config

        self.conv_in = Conv2dSame(self.config.num_channels, self.config.hidden_channels, kernel_size=3, bias=False)

        in_channel_mult = (1,) + tuple(self.config.channel_mult)
        num_res_blocks = self.config.num_res_blocks
        hidden_channels = self.config.hidden_channels

        encoder_blocks = []
        for i_level in range(self.config.num_resolutions):
            in_channels = hidden_channels * in_channel_mult[i_level]
            out_channels = hidden_channels * in_channel_mult[i_level + 1]

            if i_level < (self.config.num_resolutions - 1):
                encoder_blocks.append(DownsamplingStage(in_channels, out_channels, num_res_blocks, self.config.sample_with_conv))
            else:
                encoder_blocks.append(ResidualStage(in_channels, out_channels, num_res_blocks))
        self.down = torch.nn.ModuleList(encoder_blocks)

        # middle
        mid_channels = out_channels
        self.mid = ResidualStage(mid_channels, mid_channels, num_res_blocks)

        # end
        self.norm_out = GroupNorm(mid_channels)
        self.conv_out = Conv2dSame(mid_channels, self.config.token_size, kernel_size=1)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the convolutional encoder.

        Args:
            pixel_values -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # downsampling
        hidden_states = self.conv_in(pixel_values)
        
        for block in self.down:
            hidden_states = block(hidden_states)
        # middle
        hidden_states = self.mid(hidden_states)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class ConvDecoderLegacy(torch.nn.Module):
    """
    This is a legacy decoder class. It is used to support older weights.
    """
    def __init__(self, config):
        """ Initializes the convolutional decoder in a legacy variant.

        Args:
            config: Configuration of the model architecture.
        """
        super().__init__()

        self.config = config

        # compute in_channel_mult, block_in and curr_res at lowest res
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        num_res_blocks = self.config.num_res_blocks
        hidden_channels = self.config.hidden_channels
        in_channel_mult = tuple(self.config.channel_mult) + (self.config.channel_mult[-1],)

        # z to block_in
        self.conv_in = Conv2dSame(self.config.token_size, block_in, kernel_size=3)

        # middle
        self.mid = ResidualStage(block_in, block_in, num_res_blocks)

        # upsampling
        decoder_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            in_channels = hidden_channels * in_channel_mult[i_level + 1]
            out_channels = hidden_channels * in_channel_mult[i_level]
            if i_level > 0:
                decoder_blocks.append(UpsamplingStage(in_channels, out_channels, num_res_blocks))
            else:
                decoder_blocks.append(ResidualStage(in_channels, out_channels, num_res_blocks))

        self.up = torch.nn.ModuleList(list(reversed(decoder_blocks)))

        # end
        self.norm_out = GroupNorm(out_channels)
        self.conv_out = Conv2dSame(out_channels, self.config.num_channels, kernel_size=3)

    def forward(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the convolutional decoder.

        Args:
            z_quantized -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # z to block_in
        hidden_states = self.conv_in(z_quantized)

        # middle
        hidden_states = self.mid(hidden_states)

        # upsampling decoder
        for block in reversed(self.up):
            hidden_states = block(hidden_states, z_quantized)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class ConvDecoder(torch.nn.Module):
    def __init__(self, config):
        """ Initializes the convolutional decoder.

        Args:
            config: Configuration of the model architecture.
        """
        super().__init__()

        self.config = config

        # compute in_channel_mult, block_in and curr_res at lowest res
        block_in = self.config.hidden_channels * self.config.channel_mult[self.config.num_resolutions - 1]
        num_res_blocks = self.config.get("num_res_blocks_decoder", self.config.num_res_blocks)
        hidden_channels = self.config.hidden_channels
        in_channel_mult = tuple(self.config.channel_mult) + (self.config.channel_mult[-1],)

        # z to block_in
        if config.quantizer_type == "vae":
            self.conv_in = Conv2dSame(self.config.token_size // 2, block_in, kernel_size=3)
        else:
            self.conv_in = Conv2dSame(self.config.token_size, block_in, kernel_size=3)

        # middle
        self.mid = ResidualStage(block_in, block_in, num_res_blocks)

        # upsampling
        decoder_blocks = []
        for i_level in reversed(range(self.config.num_resolutions)):
            in_channels = hidden_channels * in_channel_mult[i_level + 1]
            out_channels = hidden_channels * in_channel_mult[i_level]
            if i_level > 0:
                decoder_blocks.append(UpsamplingStage(in_channels, out_channels, num_res_blocks))
            else:
                decoder_blocks.append(ResidualStage(in_channels, out_channels, num_res_blocks))
        self.up = torch.nn.ModuleList(decoder_blocks)

        # end
        self.norm_out = GroupNorm(out_channels)
        self.conv_out = Conv2dSame(out_channels, self.config.num_channels, kernel_size=3)

    def forward(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """ Forward pass of the convolutional decoder.

        Args:
            z_quantized -> torch.Tensor: Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # z to block_in
        hidden_states = self.conv_in(z_quantized)

        # middle
        hidden_states = self.mid(hidden_states)

        # upsampling decoder
        for block in self.up:
            hidden_states = block(hidden_states, z_quantized)

        # end
        hidden_states = self.norm_out(hidden_states)
        hidden_states = F.silu(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


if __name__ == "__main__":
    class Config:
        def __init__(self, **kwargs):
            for key in kwargs:
                setattr(self, key, kwargs[key])
        def get(self, key, default):
            return getattr(self, key, default)

    config_dict = dict(
        resolution=256,
        num_channels=3,
        hidden_channels=128,
        channel_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        codebook_size=1024,
        token_size=256,
        num_resolutions=4,
        sample_with_conv=False,
        quantizer_type="lookup",
    )
    config = Config(**config_dict)

    encoder = ConvEncoder(config)
    decoder = ConvDecoder(config)

    config.sample_with_conv = True
    encoder_conv_down = ConvEncoder(config)

    print("Encoder:\n{}".format(encoder))
    print("Encoder downsampling with conv:\n{}".format(encoder_conv_down))
    print("Decoder:\n{}".format(decoder))

    x = torch.randn((1, 3, 256, 256))
    x_enc = encoder(x)
    x_enc_down_with_conv = encoder_conv_down(x)
    x_dec = decoder(x_enc)
    x_dec_down_with_conv = decoder(x_enc_down_with_conv)

    print(f"Input shape: {x.shape}")
    print(f"Encoder output shape: {x_enc.shape}")
    print(f"Encoder with conv as down output shape: {x_enc_down_with_conv.shape}")
    print(f"Decoder output shape: {x_dec.shape}")
    print(f"Decoder with conv as down output shape: {x_dec_down_with_conv.shape}")
