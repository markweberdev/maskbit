"""This file contains the definition of utility functions for GANs."""

import torch
import torch.nn.functional as F

from modeling.modules import OriginalNLayerDiscriminator,  NLayerDiscriminatorv2


def toggle_off_gradients(model: torch.nn.Module):
    """Toggles off gradients for all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = False


def toggle_on_gradients(model: torch.nn.Module):
    """Toggles on gradients for all parameters in a model."""
    for param in model.parameters():
        param.requires_grad = True


def discriminator_weights_init(m):
    """Initialize weights for convolutions in the discriminator."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    """If global_step is less than threshold, return value, else return weight."""
    if global_step < threshold:
        weight = value
    return weight


def compute_lecam_loss(
    logits_real_mean: torch.Tensor,
    logits_fake_mean: torch.Tensor,
    ema_logits_real_mean: torch.Tensor,
    ema_logits_fake_mean: torch.Tensor
) -> torch.Tensor:
    """Computes the LeCam loss for the given average real and fake logits.

    Args:
        logits_real_mean -> torch.Tensor: The average real logits.
        logits_fake_mean -> torch.Tensor: The average fake logits.
        ema_logits_real_mean -> torch.Tensor: The EMA of the average real logits.
        ema_logits_fake_mean -> torch.Tensor: The EMA of the average fake logits.

    Returns:
        lecam_loss -> torch.Tensor: The LeCam loss.
    """
    lecam_loss = torch.mean(torch.pow(F.relu(logits_real_mean - ema_logits_fake_mean), 2))
    lecam_loss += torch.mean(torch.pow(F.relu(ema_logits_real_mean - logits_fake_mean), 2))
    return lecam_loss
  

def hinge_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Computes the hinge loss for the generator given the fake logits.

    Args:
        logits_fake -> torch.Tensor: The fake logits.

    Returns:
        g_loss -> torch.Tensor: The hinge loss.
    """
    g_loss = -torch.mean(logits_fake)
    return g_loss


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Computes the hinge loss for the discriminator given the real and fake logits.

    Args:
        logits_real -> torch.Tensor: The real logits.
        logits_fake -> torch.Tensor: The fake logits.

    Returns:
        d_loss -> torch.Tensor: The hinge loss.
    """
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def sigmoid_cross_entropy_with_logits(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """ Credits to Magvit.
    We use a stable formulation that is equivalent to the one used in TensorFlow.
    The following derivation shows how we arrive at the formulation:

    .. math::
            z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
        = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
        = (1 - z) * x + log(1 + exp(-x))
        = x - x * z + log(1 + exp(-x))

    For x < 0, the following formula is more stable:
    .. math::
            x - x * z + log(1 + exp(-x))
        = log(exp(x)) - x * z + log(1 + exp(-x))
        = - x * z + log(1 + exp(x))

    We combine the two cases (x<0, x>=0) into one formula as follows:
    .. math::
        max(x, 0) - x * z + log(1 + exp(-abs(x)))
    """
    zeros = torch.zeros_like(logits)
    cond = (logits >= zeros)
    relu_logits = torch.where(cond, logits, zeros)
    neg_abs_logits = torch.where(cond, -logits, logits)
    loss = relu_logits - logits * label + torch.log1p(neg_abs_logits.exp())
    return loss


def non_saturating_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Computes the non-saturating loss for the discriminator given the real and fake logits.
    
    Args:
        logits_real -> torch.Tensor: The real logits.
        logits_fake -> torch.Tensor: The fake logits.

    Returns:
        loss -> torch.Tensor: The non-saturating loss.
    """
    real_loss = torch.mean(
        sigmoid_cross_entropy_with_logits(logits_real, label=torch.ones_like(logits_real))
    )
    fake_loss = torch.mean(
        sigmoid_cross_entropy_with_logits(logits_fake, label=torch.zeros_like(logits_fake))
    )
    return torch.mean(real_loss) + torch.mean(fake_loss)


def non_saturating_g_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Computes the non-saturating loss for the generator given the fake logits.

    Args:
        logits_fake -> torch.Tensor: The fake logits.

    Returns:
        loss -> torch.Tensor: The non-saturating loss.
    """
    return torch.mean(
        sigmoid_cross_entropy_with_logits(logits_fake, label=torch.ones_like(logits_fake))
    )


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    """Computes the vanilla loss for the discriminator given the real and fake logits.

    Args:
        logits_real -> torch.Tensor: The real logits.
        logits_fake -> torch.Tensor: The fake logits.

    Returns:
        loss -> torch.Tensor: The vanilla loss.
    """
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) + torch.mean(torch.nn.functional.softplus(logits_fake))
    )
    return d_loss


def create_discriminator(discriminator_config) -> torch.nn.Module:
    """Creates a discriminator based on the given config.

    Args:
        discriminator_config: The config for the discriminator.

    Returns:
        discriminator -> torch.nn.Module: The discriminator.
    """
    if discriminator_config.name == "Original":
        return OriginalNLayerDiscriminator(
            num_channels=discriminator_config.num_channels,
            num_stages=discriminator_config.num_stages,
            hidden_channels=discriminator_config.hidden_channels,
        ).apply(discriminator_weights_init)
    elif discriminator_config.name == "VQGAN+Discriminator":
        return NLayerDiscriminatorv2(
            num_channels=discriminator_config.num_channels,
            num_stages=discriminator_config.num_stages,
            hidden_channels=discriminator_config.hidden_channels,
            blur_resample=discriminator_config.blur_resample,
            blur_kernel_size=discriminator_config.get("blur_kernel_size", 4)
        )
    else:
        raise ValueError(f"Discriminator {discriminator_config.name} is not implemented.")