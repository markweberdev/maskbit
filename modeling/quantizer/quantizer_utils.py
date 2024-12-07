"""This file contains the definition of some utility functions for the quantizer."""
from typing import Tuple
import torch

def clamp_log(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """ Clamps the input tensor and computes the log.

    Args:
        x -> torch.Tensor: The input tensor.
        eps -> float: The epsilon value serving as the lower bound.

    Returns:
        torch.Tensor: The log of the clamped input tensor.
    """
    return torch.log(torch.clamp(x, eps))


def entropy_loss_fn(
    affinity: torch.Tensor,
    temperature: float,
    entropy_gamma: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Computes the entropy loss.

    Args:
        affinity -> torch.Tensor: The affinity matrix.
        temperature -> float: The temperature.
        entropy_gamma -> float: The entropy gamma.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The per-sample and average entropy.
    """
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature

    probability = flat_affinity.softmax(dim=-1)
    average_probability = torch.mean(probability, dim=0)

    per_sample_entropy = -1 * torch.mean(torch.sum(probability * clamp_log(probability), dim=-1))
    avg_entropy = torch.sum(-1 * average_probability * clamp_log(average_probability))

    return (per_sample_entropy, avg_entropy * entropy_gamma)