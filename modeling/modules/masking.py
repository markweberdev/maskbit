"""This file contains the definition of utility functions for masking."""
import math
from typing import Text, Tuple
import torch


def get_mask_tokens(
    tokens: torch.Tensor,
    mask_token: int,
    mode: Text = "arccos",
    min_masking_ratio: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Get the masked tokens.
        Args:
            tokens -> torch.Tensor: The input tokens.
            mask_token -> int: The special `mask` token.
            mode -> Text: The masking function to use (default: "arccos").
        Returns:
            masked_tokens -> torch.Tensor: The masked input tokens. Each masked token is set to mask_token.
            mask -> torch.Tensor: A boolean tensor mask indicating which tokens are masked.
    """
    r = torch.rand(tokens.size(0)) * (1 - min_masking_ratio)
    if mode == "linear":
        val_to_mask = 1 - r
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.acos(r) / (math.pi * 0.5)
    else:
        raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos'.")
    
    masked_tokens = tokens.detach().clone()
    mask = torch.rand(tokens.size()) < val_to_mask.view(-1, 1, 1)

    masked_tokens[mask] = torch.full_like(masked_tokens[mask], mask_token)
    return masked_tokens, mask


def get_masking_ratio(progress: float, mode: Text = "arccos") -> torch.Tensor:
    """ Get masking ratio. 
        Args:
            progress -> float: The percentage of iterations already done.
            mode -> Text: The masking function to use (default: "arccos").

        Returns:
            val_to_mask -> torch.Tensor: The masking ratio.
    """
    r = torch.tensor(progress)
    if mode == "root":
        val_to_mask = 1 - (r ** 0.5)
    elif mode == "square":
        val_to_mask = 1 - (r ** 2)
    elif mode == "cosine":
        val_to_mask = torch.cos(r * math.pi * 0.5)
    elif mode == "arccos":
        val_to_mask = torch.acos(r) / (math.pi * 0.5)
    elif mode == "linear":
        val_to_mask = 1 - r
    else:
        raise ValueError("Invalid mode. Choose between 'linear','square', 'cosine', 'arccos', 'root'.")
    
    val_to_mask = torch.clamp(val_to_mask, 1e-6, 1.0)
    return val_to_mask