"""This file contains the definition of the look-free quantizer."""

from typing import Mapping, Text, Tuple

import torch
from einops import rearrange, reduce

from modeling.quantizer.quantizer_utils import entropy_loss_fn


class LookupFreeQuantizer(torch.nn.Module):
    def __init__(
        self,
        token_bits: int = 10,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.1,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
    ):
        """ Initializes the lookup-free quantizer.

        Args:
            token_bits -> int: The number of bits per token.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature for the entropy loss.
            entropy_gamma -> float: The gamma for the entropy loss.
        """
        super().__init__()
        self.token_size = token_bits
        self.codebook_size = 2 ** token_bits

        self.commitment_cost = commitment_cost
        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma

        bits_to_indices = torch.pow(2.0, torch.arange(0, self.token_size, dtype=torch.float32))
        self.register_buffer('bits_to_indices', bits_to_indices.int())

        all_codes = torch.arange(self.codebook_size)
        bits = ((all_codes[..., None].int() & self.bits_to_indices) != 0).float()
        self.register_buffer('codebook', bits * 2.0 - 1.0)


    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Forward pass of the quantizer.

        Args:
            z -> torch.Tensor: The input tensor.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        ones = torch.ones_like(z)
        sign_mask = (z > 0.0)
        z_quantized = torch.where(sign_mask, ones, -ones)

        min_encoding_indices = self.convert_bits_to_indices(z_quantized)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            d = - 2 * torch.einsum('b h w c, n c -> b h w n', z, self.codebook)

            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """ Returns the `codebook entry` for the given indices.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation.
        """
        indices = indices.long()
        bits = ((indices[..., None].int() & self.bits_to_indices) != 0).float()
        tokens = bits * 2.0 - 1.0  # scale to -1..1
        return tokens

    def convert_bits_to_indices(self, tokens: torch.Tensor) -> torch.Tensor:
        """ Converts the given tokens to index numbers.

        As the codebook exists only implicitly, this is mainly an integer conversion from a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            tokens -> torch.Tensor: The tokens.

        Returns:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.
        """
        tokens = rearrange(tokens, 'b h w c -> b h w c').contiguous()
        sign_mask = (tokens > 0.0)
        return reduce(sign_mask.int() * self.bits_to_indices, 'b h w c -> b h w', 'sum')

    def convert_indices_to_bits(self, indices: torch.Tensor) -> torch.Tensor:
        """ Converts the given indices to tokens.

        As the codebook exists only implicitly, this is mainly an integer conversion to a bit representation.
        Note: The bits are represented by {-1, 1}.

        Args:
            indices -> torch.Tensor: The indices in range 0 to codebook size - 1.

        Returns:
            tokens -> torch.Tensor: The bit representation.
        """
        indices = indices.long()
        return self.get_codebook_entry(indices)



if  __name__ == "__main__":
    quantizer = LookupFreeQuantizer(
        token_bits=10,
        commitment_cost=0.25,
        entropy_loss_weight=0.1,
        entropy_loss_temperature=0.01,
        entropy_gamma=1.0
    )
    all_entries = torch.arange(1024).reshape(1, 1, 1024)
    indices = quantizer.convert_bits_to_indices(quantizer.convert_indices_to_bits(all_entries))
    assert torch.equal(
        indices,
        all_entries
    )
    assert torch.equal(
        quantizer.convert_bits_to_indices(quantizer.codebook.reshape(1,1,1024,10)),
        all_entries
    )