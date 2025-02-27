# This file contains utility functions for the demo notebook.

from typing import List
import argparse
import math

import torch
import torchvision.transforms.functional as F
from omegaconf import OmegaConf
from tqdm.notebook import tqdm
from PIL import Image
from einops import rearrange

from modeling.conv_vqgan import ConvVQModel
from modeling.bert import Bert, LFQBert
from modeling.modules import sample as sample_impl


def get_config(config_path):
    conf = OmegaConf.load(config_path)
    return conf


@torch.no_grad()
def get_tokenizer(config, tokenizer_path) -> torch.nn.Module:
    tokenizer_model = ConvVQModel(config.model.vq_model, legacy=False)
    tokenizer_model.load_pretrained(tokenizer_path)
    tokenizer_model.eval()
    tokenizer_model.requires_grad_(False)
    return tokenizer_model


@torch.no_grad()
def get_generator(config, generator_path) -> torch.nn.Module:
    stage2_model_cls = {
        "bert": Bert,
        "lfq_bert": LFQBert,
    }[config.model.mlm_model.model_cls]
    
    generator_model = stage2_model_cls(
        img_size=256,
        hidden_dim=config.model.mlm_model.hidden_dim,
        codebook_size=config.model.vq_model.codebook_size,
        codebook_splits=config.model.mlm_model.codebook_splits,
        depth=config.model.mlm_model.depth,
        heads=config.model.mlm_model.heads,
        mlp_dim=config.model.mlm_model.mlp_dim,
        dropout=config.model.mlm_model.dropout,
        use_prenorm=config.model.mlm_model.use_prenorm,
        input_stride=2**(config.model.vq_model.num_resolutions - 1)
    )
    rename_dict = {"token_emb": "input_proj"}
    generator_model.load_pretrained(generator_path, rename_keys=rename_dict)
    generator_model.eval()
    generator_model.requires_grad_(False)
    return generator_model



def visualize_reconstruction(
    original_images: Image.Image,
    reconstructed_images: Image.Image,
) -> List[Image.Image]:
    """
    Creates a visualization of the samples.

    Args:
        original_images -> torch.Tensor: The original images.
        reconstructed_images -> torch.Tensor: The reconstructed images.
    
    Returns:
        A tuple of the images for saving and for logging.
    """

    difference_images = torch.abs(original_images - reconstructed_images)

    to_stack = [original_images, reconstructed_images, difference_images]

    images = rearrange(
        torch.stack(to_stack),
        "n b c h w -> b c h (n w)", b=1).byte().squeeze(0)
    images_for_saving = F.to_pil_image(images)

    return images_for_saving


def visualize_generation(
    generated_images: List[torch.Tensor],
) -> List[Image.Image]:
    """
    Creates a visualization of the samples.

    Args:
        original_images -> torch.Tensor: The original images.
        reconstructed_images -> torch.Tensor: The reconstructed images.
    
    Returns:
        A tuple of the images for saving and for logging.
    """
    images = rearrange(
        torch.stack(generated_images),
        "n b c h w -> n c h (b w)").byte()
    images_for_saving = [F.to_pil_image(image) for image in images]

    return images_for_saving


def sample(
    generator,
    tokenizer,
    config,
    labels,
    samples_per_class,
    guidance_scale,
    temperature,
    steps,
    device,
):
    if config.model.vq_model.quantizer_type == "lookup-free":
        num_codebook_entries = 2 ** config.model.vq_model.token_size
        mask_token = int(2 ** (math.log2(num_codebook_entries) // config.model.mlm_model.codebook_splits))
    else:
        num_codebook_entries = config.model.vq_model.codebook_size
        mask_token = int(2 ** (math.log2(num_codebook_entries) // config.model.mlm_model.codebook_splits))
    
    with torch.no_grad():

        tokenizer.eval()
        generator.eval()

        generated_list = []
        batchsize = samples_per_class


        print("Running generation...")
        for i in tqdm(range(labels.shape[0]), desc="Generating samples", position=0):
            y = labels[i].repeat(samples_per_class)

            generated_samples, _ = sample_impl(
                generator,
                tokenizer,
                num_samples=batchsize,
                labels=y,
                softmax_temperature=1.0,
                randomize_temperature=temperature,
                mask_schedule_strategy="arccos",
                num_steps=steps,
                guidance_scale=guidance_scale,
                mask_token=mask_token,
                patch_size=int(256 // 2**(config.model.vq_model.num_resolutions - 1)),
                guidance_annealing="none",
                use_sampling_annealing=False,
                scale_pow=1.0,
                codebook_size=config.model.vq_model.codebook_size,
                codebook_splits=config.model.mlm_model.codebook_splits,
                use_tqdm=False,
            )
                
            generated_samples = torch.clamp(generated_samples, 0.0, 1.0)
            generated_samples = (generated_samples * 255.0).cpu()

            generated_list.append(generated_samples)

    
    return generated_list