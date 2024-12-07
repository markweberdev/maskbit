"""This file contains a script to evaluate a tokenizer."""

# Eval script for VQGAN
import sys

import os
from pathlib import Path
import pprint

import tqdm

from data import SimpleImagenet
import torch
import torchvision
from omegaconf import OmegaConf

from utils.logger import setup_logger
from modeling.conv_vqgan import ConvVQModel
from modeling.taming_vqgan import OriginalVQModel
from evaluator import TokenizerEvaluator

logger = setup_logger(name="MaskBit", log_level="INFO", use_accelerate=False)


def get_config():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def main():

    output_dir = None

    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


    #########################
    # MODELS                #
    #########################
    logger.info("Creating model")

    if config.model.vq_model.quantizer_type == "lookup-free":
        num_codebook_entries = 2 ** config.model.vq_model.token_size
        config.model.vq_model.codebook_size = num_codebook_entries

    if config.model.vq_model.model_class not in ("taming", "maskgit", "vqgan+"):
        raise ValueError(f"Tokenizer model must be one of `taming`, `maskgit`, `vqgan+`."
                         "{config.model.vq_model.model_class} is not supported.")

    if config.model.vq_model.model_class == "vqgan+":
        model = ConvVQModel(config.model.vq_model)
    elif config.model.vq_model.model_class == "taming":
        model = OriginalVQModel(config.model.vq_model)
    else:
         model = ConvVQModel(config.model.vq_model, legacy=True)

    model.load_pretrained(config.experiment.vqgan_checkpoint)
    model.to("cuda:0")

    print(f"Evaluating tokenizer model at {config.dataset.preprocessing.resolution} resolution.")

    ##################################
    # DATLOADER                      #
    ##################################
    logger.info("Creating dataloaders")

    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset = SimpleImagenet(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=config.training.per_gpu_batch_size,
        num_workers_per_gpu=dataset_config.num_workers_per_gpu,
        resolution=preproc_config.resolution,
        shuffle_buffer_size=dataset_config.shuffle_buffer_size,
        pin_memory=dataset_config.pin_memory,
        persistent_workers=dataset_config.persistent_workers,
        use_aspect_ratio_aug=preproc_config.use_aspect_ratio_aug,
        use_random_crop=preproc_config.use_random_crop,
        min_scale=preproc_config.min_scale,
        interpolation=preproc_config.interpolation,
    )
    eval_dataloader = dataset.eval_dataloader

    ##################################
    # EVALUATION STUFF.              #
    ##################################
    evaluator = TokenizerEvaluator(
        device="cuda:0",
        enable_rfid=True,
        enable_inception_score=True,
        enable_psnr_score=True,
        enable_ssim_score=True,
        enable_lpips_score=True,
        enable_mse_error=True,
        enable_mae_error=True,
        enable_codebook_usage_measure=True,
        enable_codebook_entropy_measure=True,
        num_codebook_entries=config.model.vq_model.codebook_size
    )

    eval_scores = eval_reconstruction(
        model,
        eval_dataloader,
        evaluator,
        output_dir=output_dir
    )

    logger.info(f"EVALUATION")
    logger.info(pprint.pformat(eval_scores))


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    evaluator,
    output_dir = None
):
    model.eval()
    evaluator.reset_metrics()


    for batch in tqdm.tqdm(eval_loader):
        images = batch["image"].to(
            "cuda:0", memory_format=torch.contiguous_format, non_blocking=True
        )
        fnames = batch["__key__"]

        original_images = torch.clone(images)
        reconstructed_images, model_dict = model(images)
        
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        original_images = torch.clamp(original_images, 0.0, 1.0)

        evaluator.update(original_images, reconstructed_images, model_dict["min_encoding_indices"])

        if output_dir is not None:
            root = Path(output_dir) / "eval_images"
            os.makedirs(root, exist_ok=True)
            os.makedirs(root / "reconstructed", exist_ok=True)
            os.makedirs(root / "original", exist_ok=True)
            os.makedirs(root, exist_ok=True)
            for i,(orig_img, rec_img) in enumerate(zip(original_images, reconstructed_images)):
                filename = f"{fnames[i]}.png"
                path = os.path.join(root / "original", filename)
                orig_img = torchvision.transforms.functional.to_pil_image(orig_img)
                orig_img.save(path)
                path = os.path.join(root / "reconstructed", filename)
                rec_img = torchvision.transforms.functional.to_pil_image(rec_img)
                rec_img.save(path)


    return evaluator.result()


if __name__ == "__main__":
    main()