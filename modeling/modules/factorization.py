"""This file contains the definition of utility functions to group tokens."""

import math
import torch


def combine_factorized_tokens(tokens: torch.Tensor, codebook_size: int, splits: int) -> torch.Tensor:
    """
    Combine the tokens into a single token.

    Args:
        tokens -> torch.Tensor: Tensor of shape (batch_size, n, m).
        codebook_size -> int: The size of the codebook.
        splits -> int: Number of splits.
    
    Returns:
        combined_tokens -> torch.Tensor: Tensor of shape (batch_size, n).
    """
    combined_tokens = torch.zeros((tokens.shape[0], tokens.shape[1]), device=tokens.device)
    bit_shift = int(math.log2(codebook_size)) // splits
    for i in range(splits):
        combined_tokens += (tokens[..., i] << (i * bit_shift))

    return combined_tokens


def split_factorized_tokens(tokens: torch.Tensor, codebook_size: int, splits: int) -> torch.Tensor:
    """
    Split the tokens into multiple tokens.

    Args:
        tokens -> torch.Tensor: Tensor of shape (batch_size, n).
        codebook_size -> int: The size of the codebook.
        splits -> int: Number of splits.
    
    Returns:
        split_tokens -> torch.Tensor: Tensor of shape (batch_size, n, m).
    """
    bit_shift = int(math.log2(codebook_size)) // splits
    bit_mask = (1 << bit_shift) - 1

    split_tokens = []
    for i in range(splits):
        split_tokens.append((tokens & (bit_mask << (i * bit_shift))) >> (i * bit_shift))

    return torch.stack(split_tokens, dim=2)


if __name__ == "__main__":
    tokens = torch.randint(0, 1023, (1, 16))
    split_tokens = split_factorized_tokens(tokens, 1024, 1)

    assert split_tokens.shape == (1, 16, 1)
    assert split_tokens.dtype == torch.int64

    combined_tokens = combine_factorized_tokens(split_tokens, 1024, 1)

    assert (tokens == combined_tokens).all()

    split_tokens = split_factorized_tokens(tokens, 1024, 2)
    combined_tokens = combine_factorized_tokens(split_tokens, 1024, 2)

    assert split_tokens.shape == (1, 16, 2)
    assert (tokens == combined_tokens).all(), f"{tokens} != {combined_tokens}"

    assert (torch.bitwise_right_shift(tokens, 5) == split_tokens[..., 1]).all()
    assert (tokens & 31 == split_tokens[..., 0]).all()