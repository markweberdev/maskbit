""" This file contains some utils functions for visualization."""

from typing import Tuple, List

import torch
import torchvision.transforms.functional as F

from PIL import Image
from einops import rearrange


def make_viz_from_samples(
    original_images: torch.Tensor,
    reconstructed_images: torch.Tensor,
) -> Tuple[List[Image.Image], torch.Tensor]:
    """
    Creates a visualization of the samples.

    Args:
        original_images -> torch.Tensor: The original images.
        reconstructed_images -> torch.Tensor: The reconstructed images.
    
    Returns:
        A tuple of the images for saving and for logging.
    """

    # Convert to PIL images
    reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
    original_images = torch.clamp(original_images, 0.0, 1.0)
    reconstructed_images *= 255.0
    original_images *= 255.0
    reconstructed_images = reconstructed_images.cpu()
    original_images = original_images.cpu()
    difference_images = torch.abs(original_images - reconstructed_images)

    to_stack = [original_images, reconstructed_images, difference_images]

    images_for_logging = rearrange(
        torch.stack(to_stack),
        "(l1 l2) b c h w -> b c (l1 h) (l2 w)",
        l1=1).byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_reconstructed_stage_two(
    reconstructed: torch.Tensor,
    predicted: torch.Tensor
) -> Tuple[List[Image.Image], torch.Tensor]:
    """
    Creates a visualization of the reconstructed and predicted samples.

    Args:
        predicted -> torch.Tensor: The predicted images.
        reconstructed -> torch.Tensor: The reconstructed images.
    Returns:
        A tuple of the images for saving and for logging.
    """
    reconstructed = torch.clamp(reconstructed, 0.0, 1.0)
    predicted = torch.clamp(predicted, 0.0, 1.0)

    images_for_logging = torch.cat(
        [reconstructed, predicted], dim=-1
    ) * 255.0
    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = [F.to_pil_image(image) for image in images_for_logging]

    return images_for_saving, images_for_logging


def make_viz_generated_stage_two(
    generated: torch.Tensor,
)-> Tuple[Image.Image, torch.Tensor]:
    """
    Creates a visualization of the generated samples.

    Args:
        generated -> torch.Tensor: The generated images, shape (batch_size, 3, height, width), 
            where batch_size is an even number.
    Returns:
        A tuple of the images for saving and for logging.
    """
    generated = torch.clamp(generated, 0.0, 1.0) * 255.0
    images_for_logging = rearrange(
        generated, 
        "(l1 l2) c h w -> c (l1 h) (l2 w)",
        l1=2)

    images_for_logging = images_for_logging.cpu().byte()
    images_for_saving = F.to_pil_image(images_for_logging)

    return images_for_saving, images_for_logging
