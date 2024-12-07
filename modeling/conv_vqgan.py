"""This file contains the definition of the our tokenizer, which can use VQ or LFQ."""

import math
from typing import Mapping, Text, Tuple

import torch
from einops import rearrange

from modeling.modules import (BaseModel, ConvDecoder, ConvDecoderLegacy,
                              ConvEncoder)
from modeling.quantizer import LookupFreeQuantizer, SimpleVectorizer


def choose_vector_quantizer_class(config):
    if config.quantizer_type == "lookup":
        return SimpleVectorizer(
            config.codebook_size,
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
            config.get("use_l2_normalisation", False),
        )
    elif config.quantizer_type == "lookup-free":
        return LookupFreeQuantizer(
            config.token_size,
            config.commitment_cost,
            config.entropy_loss_weight,
            config.entropy_loss_temperature,
            config.entropy_gamma,
        )
    elif config.quantizer_type == "vae":
        return NotImplementedError("Currently not supported. We welcome a well tested PR.")
    else:
        raise ValueError("Unknown vector quantizer class")


class ConvVQModel(BaseModel):
    def __init__(
        self,
        config,
        legacy: bool = False,
        finetune_decoder: bool = False
    ):
        """ Initializes the convolutional VQ-VAE model.

        Args:
            config: The configuration for the model.
            legacy -> bool: Whether to use the legacy decoder, which is a different implementation of the same architecture.
            finetune_decoder -> bool: Whether to finetune the decoder.
        """
        super().__init__()
        self.config = config
        self.encoder = ConvEncoder(self.config)
        if legacy:
            # To support older weights and MaskGIT
            self.decoder = ConvDecoderLegacy(self.config)
        else:
            self.decoder = ConvDecoder(self.config)

        self.finetune_decoder = finetune_decoder
        if self.finetune_decoder:
            self.encoder.eval()
            self.encoder.requires_grad_(False)
        self.quantize = choose_vector_quantizer_class(self.config)

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Encodes the input tensor, i.e. runs the encoder.

        Args:
            x -> torch.Tensor: The input tensor.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = self.encoder(x)
        z_quantized, result_dict = self.quantize(z)
        return z_quantized, result_dict

    def decode(self, z_quantized: torch.Tensor) -> torch.Tensor:
        """ Decodes the quantized latent representation, i.e. runs the decoder.

        Args:
            z_quantized -> torch.Tensor: The quantized latent representation.

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        decoded = self.decoder(z_quantized)
        return decoded

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Decodes from tokens, i.e. runs the decoder after converting tokens to latent representations.

        Args:
            tokens -> torch.Tensor: The tokens.

        Returns:
            decoded -> torch.Tensor: The decoded image.
        """
        z_quantized = self.quantize.get_codebook_entry(tokens)
        ss = int(math.sqrt(float(z_quantized.size(1))))
        z_quantized = z_quantized.reshape(z_quantized.size(0), ss, ss, -1)
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()
        decoded = self.decode(z_quantized)
        return decoded

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Runs the model on the input tensor.

        Args:
            input -> torch.Tensor: The input image.

        Returns:
            decoded -> torch.Tensor: The decoded image.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        if self.finetune_decoder:
            self.encoder.eval()
            z_quantized, result_dict = self._finetuning_encoder_forward(input)
        else:
            z_quantized, result_dict = self.encode(input)

        decoded = self.decode(z_quantized)
        return decoded, result_dict

    def _finetuning_encoder_forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Runs the encoder on the input tensor without gradients and sets quantizer losses to 0.

        Args:
            input -> torch.Tensor: The input image.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        with torch.no_grad():
            z_quantized, result_dict = self.encode(input)
            result_dict["quantizer_loss"] *= 0
            result_dict["commitment_loss"] *= 0
            if "codebook_loss" in result_dict:
                result_dict["codebook_loss"] *= 0
            result_dict["entropy_loss"] *= 0
        return z_quantized, result_dict