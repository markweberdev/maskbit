"""This file contains the definition of the VQ quantizer."""
from typing import Mapping, Text, Tuple

import torch
from einops import rearrange

from modeling.quantizer.quantizer_utils import entropy_loss_fn


class SimpleVectorizer(torch.nn.Module):
    def __init__(
        self,
        codebook_size: int = 1024,
        token_size: int = 256,
        commitment_cost: float = 0.25,
        entropy_loss_weight: float = 0.0,
        entropy_loss_temperature: float = 0.01,
        entropy_gamma: float = 1.0,
        use_l2_normalisation: bool = False,
    ):
        """ Initializes the quantizer.

        Args:
            codebook_size -> int: The size of the codebook.
            token_size -> int: The feature dimensions of the tokens.
            commitment_cost -> float: The commitment cost.
            entropy_loss_weight -> float: The weight of the entropy loss.
            entropy_loss_temperature -> float: The temperature of the entropy loss.
            entropy_gamma -> float: The gamma of the entropy loss.
            use_l2_normalisation -> bool: Whether to use L2 normalisation.
        """

        super().__init__()
        self.commitment_cost = commitment_cost

        self.embedding = torch.nn.Embedding(codebook_size, token_size)
        self.embedding.weight.data.uniform_(-1.0 / codebook_size, 1.0 / codebook_size)

        self.entropy_loss_weight = entropy_loss_weight
        self.entropy_loss_temperature = entropy_loss_temperature
        self.entropy_gamma = entropy_gamma
        self.use_l2_normalisation = use_l2_normalisation

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Mapping[Text, torch.Tensor]]:
        """ Computes the quantization loss and returns the quantized latent representation.

        Args:
            z -> torch.Tensor: The latent representation.

        Returns:
            z_quantized -> torch.Tensor: The quantized latent representation.
            result_dict -> Mapping[Text, torch.Tensor]: A dictionary containing additional results
                and losses from the quantizer.
        """
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()

        if self.use_l2_normalisation:
            z = torch.nn.functional.normalize(z, dim=-1)
            embedding = torch.nn.functional.normalize(self.embedding.weight, dim=-1)
        else:
            embedding = self.embedding.weight

        z_flattened = rearrange(z, 'b h w c -> (b h w) c')

        # distances from z to embeddings e_j d = (z - e)^2 = z^2 + e^2 - 2 e * z
        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + \
            torch.sum(embedding**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, embedding.T)

        min_encoding_indices = torch.argmin(d, dim=1)
        z_quantized = self.get_codebook_entry(min_encoding_indices).view(z.shape)

        # compute loss for embedding
        commitment_loss = self.commitment_cost * torch.mean((z_quantized.detach() - z) **2)
        codebook_loss = torch.mean((z_quantized - z.detach()) **2)
        entropy_loss = torch.zeros((), device=z.device)
        per_sample_entropy = torch.zeros((), device=z.device)
        avg_entropy = torch.zeros((), device=z.device)

        # Use entropy loss on the codebook
        if self.entropy_loss_weight != 0.0 and self.training:
            per_sample_entropy, avg_entropy = entropy_loss_fn(-1*d, self.entropy_loss_temperature, self.entropy_gamma)
            entropy_loss = self.entropy_loss_weight * (per_sample_entropy - avg_entropy)

        loss = commitment_loss + codebook_loss + entropy_loss

        # preserve gradients
        z_quantized = z + (z_quantized - z).detach()

        # reshape back to match original input shape
        z_quantized = rearrange(z_quantized, 'b h w c -> b c h w').contiguous()

        result_dict = dict(
            quantizer_loss=loss,
            commitment_loss=commitment_loss,
            codebook_loss=codebook_loss,
            entropy_loss=entropy_loss,
            per_sample_entropy=per_sample_entropy,
            avg_entropy=avg_entropy,
            min_encoding_indices=min_encoding_indices.view(z_quantized.shape[0], z_quantized.shape[2], z_quantized.shape[3])
        )

        return z_quantized, result_dict

    def get_codebook_entry(self, indices: torch.Tensor) -> torch.Tensor:
        """ Returns the codebook entry for the given indices.
        
        Args:
            indices -> torch.Tensor: The indices of the codebook entries.

        Returns:
            z_quantized -> torch.Tensor: The codebook entries.
        """
        # get quantized latent vectors
        z_quantized = self.embedding(indices.int())
        if self.use_l2_normalisation:
            z_quantized = torch.nn.functional.normalize(z_quantized, dim=-1)
        return z_quantized
