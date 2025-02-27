"""This file contains the definition of the sampling function."""

from typing import Optional, Tuple, List, Text
import tqdm

import torch

from .masking import get_masking_ratio
from .factorization import combine_factorized_tokens


@torch.no_grad()
def sample(
    model,
    vqgan_model,
    num_samples: int = 10,
    labels: Optional[torch.Tensor] = None,
    softmax_temperature: float = 1.0,
    randomize_temperature: float = 4.5,
    mask_schedule_strategy: Text = "linear",
    num_steps: int = 12,
    guidance_scale: float = 3.0,
    mask_token: int = 1024,
    patch_size: int = 16,
    guidance_annealing: Text = "none",
    use_sampling_annealing: bool = False,
    scale_pow: float = 4.0,
    codebook_size: int = 1024,
    codebook_splits: int = 1,
    use_tqdm: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Sample from the model.

    Args:
        model -> torch.nn.Module: The model to sample from.
        vqgan_model -> torch.nn.Module: The VQGAN model.
        num_samples -> int: The number of samples to generate.
        labels -> Optional[torch.Tensor]: The labels to use for the generation.
        softmax_temperature -> float: The temperature for the softmax.
        randomize_temperature -> float: The temperature for the randomization.
        mask_schedule_strategy -> Text: The strategy for the mask schedule.
        num_steps -> int: The number of steps to use for the sampling.
        guidance_scale -> float: The scale for the guidance.
        mask_token -> int: The token to use for the masking.
        patch_size -> int: The size of the patches.
        guidance_annealing -> Text: The annealing strategy for the guidance.
        use_sampling_annealing -> bool: Whether to use the sampling annealing.
        scale_pow -> float: The power for the scaling.
        codebook_size -> int: The size of the codebook.
        codebook_splits -> int: The number of splits for the codebook.

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor]]: The generated samples and the tokens at each step.
    """
    device = model.device

    model.eval()
    vqgan_model.eval()

    if labels is None:
        # goldfish, chicken, tiger cat, hourglass, ship, dog, race car, airliner, teddy bear, random
        labels = [1, 7, 282, 604, 724, 179, 751, 404, 850, torch.randint(0, 999, size=(1,))] * (num_samples // 10)
        labels = torch.LongTensor(labels).to(device)

    drop_labels = torch.ones(num_samples, dtype=bool, device=device)
    spatial_size = int(patch_size ** 2)
    num_splits = int(codebook_splits)
    
    masked_tokens = torch.full((num_samples, spatial_size, num_splits), mask_token, device=device)
    num_maskable = spatial_size * num_splits
    mask = (masked_tokens == mask_token)

    l_full_tokens = []
    gumbel = torch.distributions.Gumbel(loc=0.0, scale=1.0)

    if use_tqdm:
        step_iterable = tqdm.tqdm(range(num_steps), desc="Sampling steps", position=1)
    else:
        step_iterable = range(num_steps)

    for i in step_iterable:
        progress = (i + 1) / num_steps
        if guidance_scale != 0.0:
            logits = model(
                torch.cat([masked_tokens.clone(), masked_tokens.clone()], dim=0),
                torch.cat([labels, labels], dim=0),
                torch.cat([~drop_labels, drop_labels], dim=0)
            )
            # Classifier-free guidance
            logits_with_class, logits_without_class = torch.chunk(logits, 2, dim=0)
            if guidance_annealing == "none":
                scale_step = 1.0
            elif guidance_annealing == "linear":
                scale_step = i / num_steps
            elif guidance_annealing == "cosine":
                scale_pow = torch.ones((1), device=device) * scale_pow
                scale_step = (1 - torch.cos(((i / num_steps) ** scale_pow) * torch.pi)) * 1/2 # power-cos scaling
            scale = guidance_scale * scale_step
            logits = logits_with_class + scale * (logits_with_class - logits_without_class)
        else:
            logits = model(masked_tokens.clone(), labels, ~drop_labels)
        
        if use_sampling_annealing:
            softmax_temperature = 0.5 + 0.8 * (1 - progress)
        probabilities = torch.softmax(logits / softmax_temperature, dim=-1)
        distribution = torch.distributions.Categorical(probabilities)
        predicted_tokens = distribution.sample()

        num_masked = torch.sum(mask, dim=(1,2))[0]

        predicted_tokens = torch.where(mask, predicted_tokens, masked_tokens)

        confidence = torch.gather(probabilities, -1, predicted_tokens.unsqueeze(-1)).squeeze(-1)
        # Ignore existing tokens by overwriting the confidence.
        confidence = torch.where(mask, confidence, torch.inf)

        noise = gumbel.sample(predicted_tokens.size()) * randomize_temperature * (1 - progress)
        confidence = torch.log(confidence) + noise.to(device)

        mask_ratio = get_masking_ratio(progress, mode=mask_schedule_strategy).to(device)
        
        # min = 1, max = num_masked - 1
        mask_len = torch.floor(mask_ratio * num_maskable)
        num_tokens_to_mask = torch.clamp(mask_len, torch.ones_like(num_masked), num_masked-1).long()
        sorted_confidence = torch.sort(confidence.view(num_samples, -1), dim=-1).values
        threshold = sorted_confidence[:, num_tokens_to_mask - 1]

        should_mask = (confidence <= threshold.unsqueeze(-1).unsqueeze(-1))
        masked_tokens = torch.where(should_mask, mask_token, predicted_tokens)
        mask = (masked_tokens == mask_token)
        l_full_tokens.append(predicted_tokens)

    predicted_tokens = combine_factorized_tokens(predicted_tokens, codebook_size, codebook_splits)

    generated_image = vqgan_model.decode_tokens(predicted_tokens)
    return generated_image, l_full_tokens