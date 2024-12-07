"""This file contains the training script for the tokenizer."""

import json
import math
import os
import time
from pathlib import Path
from collections import defaultdict
import pprint
import glob

from accelerate.utils import DistributedType, set_seed
from accelerate import Accelerator
from accelerate.logging import get_logger

from data import SimpleImagenet
import torch
from omegaconf import OmegaConf
from torch.optim import AdamW
from utils.lr_schedulers import get_scheduler
from utils.logger import setup_logger
from utils.meter import AverageMeter
from modeling.modules import EMAModel, VQGANLoss
from modeling.conv_vqgan import ConvVQModel
from evaluator import TokenizerEvaluator

from utils.viz_utils import make_viz_from_samples

from torchinfo import summary


def get_config():
    cli_conf = OmegaConf.from_cli()

    yaml_conf = OmegaConf.load(cli_conf.config)
    conf = OmegaConf.merge(yaml_conf, cli_conf)

    return conf


def main():
    #########################
    # SETUP Accelerator     #
    #########################
    ckpt_dir = os.environ.get('WORKSPACE', './runs')

    config = get_config()

    # Enable TF32 on Ampere GPUs
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    output_dir = os.path.join(ckpt_dir, "outputs", config.experiment.name)
    config.experiment.logging_dir = str(Path(output_dir) / "logs")

    if config.experiment.logger not in ("wandb", "tensorboard"):
        raise ValueError(f"{config.experiment.logger} is not supported. Please choose `wandb` or `tensorboard`.")

    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=config.experiment.logger,
        project_dir=config.experiment.logging_dir,
        split_batches=False,
    )

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = (
            config.training.per_gpu_batch_size
        )

    #####################################
    # SETUP LOGGING, SEED and CONFIG    #
    #####################################
    if accelerator.is_local_main_process:
        os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(name="VQGAN", log_level="INFO", output_dir=config.experiment.logging_dir)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(config.experiment.name)
        config_path = Path(output_dir) / "config.yaml"
        logger.info(f"Saving config to {config_path}")
        OmegaConf.save(config, config_path)
    
    if accelerator.is_local_main_process:
        logger.info(f"Config:\n{OmegaConf.to_yaml(config)}", main_process_only=False)

    # If passed along, set the training seed now.
    if config.training.seed is not None:
        set_seed(config.training.seed)

    #########################
    # MODELS and OPTIMIZER  #
    #########################
    logger.info("Creating model and optimizer")

    model = ConvVQModel(config.model.vq_model)

    # Create the EMA model
    if config.training.use_ema:
        ema_model = EMAModel(model.parameters(), decay=0.999, model_cls=ConvVQModel, config=config.model.vq_model)

        # Create custom saving and loading hooks so that `accelerator.save_state(...)` serializes in a nice format.
        def load_model_hook(models, input_dir):
            load_model = EMAModel.from_pretrained(os.path.join(input_dir, "ema_model"), model_cls=ConvVQModel, config=config.model.vq_model)
            ema_model.load_state_dict(load_model.state_dict())
            ema_model.to(accelerator.device)
            del load_model

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                ema_model.save_pretrained(os.path.join(output_dir, "ema_model"))

        accelerator.register_load_state_pre_hook(load_model_hook)
        accelerator.register_save_state_pre_hook(save_model_hook)

    loss_config = config.losses
    loss_module = VQGANLoss(
        config.model.discriminator,
        loss_config=loss_config
    )

    # Print Model:
    input_size = (1, 3, config.dataset.preprocessing.resolution, config.dataset.preprocessing.resolution)
    model_summary_str = summary(
        model,
        input_size=input_size,
        depth=5,
        col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"),
        verbose=0
    )
    discriminator_summary_str = summary(
        loss_module.discriminator,
        input_size=input_size,
        depth=3,
        col_names=("input_size", "output_size", "num_params", "params_percent", "kernel_size", "mult_adds"),
        verbose=0
    )
    logger.info(model_summary_str)
    logger.info(discriminator_summary_str)


    optimizer_config = config.optimizer.params
    learning_rate = optimizer_config.learning_rate
    discriminator_learning_rate = optimizer_config.discriminator_learning_rate
    if optimizer_config.scale_lr:
        learning_rate = (
            learning_rate
            * config.training.per_gpu_batch_size
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )
        discriminator_learning_rate = (
            discriminator_learning_rate
            * config.training.per_gpu_batch_size
            * accelerator.num_processes
            * config.training.gradient_accumulation_steps
        )

    optimizer_type = config.optimizer.name
    if optimizer_type == "adamw":
        optimizer_cls = AdamW
    else:
        raise ValueError(f"Optimizer {optimizer_type} not supported")

    optimizer = optimizer_cls(
        list(model.parameters()),
        lr=learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )
    discriminator_optimizer = optimizer_cls(
        list(loss_module.parameters()),
        lr=discriminator_learning_rate,
        betas=(optimizer_config.beta1, optimizer_config.beta2),
        weight_decay=optimizer_config.weight_decay,
        eps=optimizer_config.epsilon,
    )

    ##################################
    # DATLOADER and LR-SCHEDULER     #
    ##################################
    logger.info("Creating dataloaders and lr_scheduler")

    total_batch_size_without_accum = config.training.per_gpu_batch_size * accelerator.num_processes
    total_batch_size = (
        config.training.per_gpu_batch_size * accelerator.num_processes * config.training.gradient_accumulation_steps
    )

    lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
    )
    discriminator_lr_scheduler = get_scheduler(
        config.lr_scheduler.scheduler,
        optimizer=discriminator_optimizer,
        num_training_steps=config.training.max_train_steps * accelerator.num_processes - config.losses.discriminator_start,
        num_warmup_steps=config.lr_scheduler.params.warmup_steps * accelerator.num_processes,
    )

    # DataLoaders creation:
    # We use webdataset for data loading. The dataloaders are created with sampling with replacement.
    # We don't do dataset resuming here, instead we resample the shards and buffer each time. The sampling is stochastic.
    # This means that the dataloading is not deterministic, but it's fast and efficient.
    preproc_config = config.dataset.preprocessing
    dataset_config = config.dataset.params

    dataset = SimpleImagenet(
        train_shards_path=dataset_config.train_shards_path_or_url,
        eval_shards_path=dataset_config.eval_shards_path_or_url,
        num_train_examples=config.experiment.max_train_examples,
        per_gpu_batch_size=config.training.per_gpu_batch_size,
        global_batch_size=total_batch_size_without_accum,
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
    train_dataloader, eval_dataloader = dataset.train_dataloader, dataset.eval_dataloader

    ##################################
    # EVALUATION STUFF.              #
    ##################################
    if config.model.vq_model.quantizer_type == "lookup-free":
        num_codebook_entries = 2 ** config.model.vq_model.token_size
        config.model.vq_model.codebook_size = num_codebook_entries
    else:
        num_codebook_entries = config.model.vq_model.codebook_size
    evaluator = TokenizerEvaluator(
        device=accelerator.device,
        enable_rfid=True,
        enable_inception_score=True,
        enable_psnr_score=True,
        enable_ssim_score=True,
        enable_lpips_score=True,
        enable_mse_error=True,
        enable_mae_error=True,
        enable_codebook_usage_measure=True,
        enable_codebook_entropy_measure=True,
        num_codebook_entries=num_codebook_entries
    )

    # Prepare everything with accelerator
    logger.info("Preparing model, optimizer and dataloaders")
    # The dataloader are already aware of distributed training, so we don't need to prepare them.
    model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler = accelerator.prepare(
        model, loss_module, optimizer, discriminator_optimizer, lr_scheduler, discriminator_lr_scheduler
    )
    if config.training.use_ema:
        ema_model.to(accelerator.device)

    if config.training.overfit_batch:
        num_update_steps_per_epoch = config.training.overfit_batch_num
        new_train_dataloader = []
        for i, batch in enumerate(train_dataloader):
            if i >= num_update_steps_per_epoch:
                break
            new_train_dataloader.append(batch)
        train_dataloader = new_train_dataloader
        logger.info(f"Overfitting on {len(train_dataloader)} batch(es).")
    else:
        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(train_dataloader.num_batches / config.training.gradient_accumulation_steps)
    
    # Afterwards we recalculate our number of training epochs.
    # Note: We are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    num_train_epochs = math.ceil(config.training.max_train_steps / num_update_steps_per_epoch)

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Num training steps = {config.training.max_train_steps}")
    logger.info(f"  Gradient Accumulation steps = {config.training.gradient_accumulation_steps}")
    logger.info(f"  Instantaneous batch size per gpu = { config.training.per_gpu_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    global_step = 0
    first_epoch = 0

    if config.experiment.init_checkpoint != '' and os.path.exists(config.experiment.init_checkpoint):
        global_step = load_checkpoint(Path(config.experiment.init_checkpoint), accelerator)

        if config.training.use_ema:
            ema_model.set_step(global_step)

    if config.experiment.resume:
        accelerator.wait_for_everyone()
        local_ckpt_list = list(glob.glob(os.path.join(output_dir, "checkpoint*")))
        if len(local_ckpt_list) >= 1:
            if len(local_ckpt_list) > 1:
                fn = lambda x: int(x.split('/')[-1].split('-')[-1])
                checkpoint_paths = sorted(local_ckpt_list, key=fn, reverse=True)
            else:  # len(local_ckpt_list) == 1
                checkpoint_paths = local_ckpt_list
            
            resume_lr_scheduler = config.experiment.get("resume_lr_scheduler", True)
            dont_resume_optimizer = config.experiment.get("dont_resume_optimizer", False)
            if not resume_lr_scheduler:
                logger.info("Not resuming the lr scheduler.")
                accelerator._schedulers = []  # very hacky, but we don't want to resume the lr scheduler
            if dont_resume_optimizer:
                logger.info("Not resuming the optimizer.")
                accelerator._optimizers = []  # very hacky, but we don't want to resume the optimizer
                grad_scaler = accelerator.scaler
                accelerator.scaler = None

            global_step = load_checkpoint(
                Path(checkpoint_paths[0]),
                accelerator
            )
            if config.training.use_ema:
                ema_model.set_step(global_step)
            if not resume_lr_scheduler:
                accelerator._schedulers = [lr_scheduler]
            if dont_resume_optimizer:
                accelerator._optimizers = [optimizer]
                accelerator.scaler = grad_scaler

            first_epoch = global_step // num_update_steps_per_epoch
        else:
            logger.info("Training from scratch.")

    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()
    # As stated above, we are not doing epoch based training here, but just using this for book keeping and being able to
    # reuse the same training loop with other datasets/loaders.
    for current_epoch in range(first_epoch, num_train_epochs):
        model.train()
        accelerator.print(f"Epoch {current_epoch}/{num_train_epochs-1} started.")
        for batch in train_dataloader:
            images = batch["image"].to(
                accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
            )
            fnames = batch["__key__"]
            data_time_m.update(time.time() - end)

            with accelerator.accumulate([model, loss_module]):
                reconstructed_images, extra_results_dict = model(images)

                # ########################
                # autoencoder loss
                # ########################
                autoencoder_loss, loss_dict = loss_module(
                    images,
                    reconstructed_images,
                    extra_results_dict,
                    global_step,
                    last_layer=accelerator.unwrap_model(model).get_last_layer(),
                    mode="gen",
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                autoencoder_logs = {}
                for k, v in loss_dict.items():
                    if k in ["discriminator_factor", "d_weight"]:  # Unclear why gan loss and perceptual loss should not be gathered "train/g_loss", "train/p_loss"]:
                        if type(v) == torch.Tensor:
                            autoencoder_logs["train/" + k] = v.cpu().item()
                        else:
                            autoencoder_logs["train/" + k] = v
                    else:
                        autoencoder_logs["train/" + k] = accelerator.gather(v).mean().item()

                accelerator.backward(autoencoder_loss)

                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()

                # log gradient norm before zeroing it
                if (
                    accelerator.sync_gradients
                    and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                    and accelerator.is_main_process
                ):
                    log_grad_norm(model, accelerator, global_step + 1)

                optimizer.zero_grad(set_to_none=True)
                
                # ########################
                # discriminator loss
                # ########################
                discriminator_logs = defaultdict(float)
                if accelerator.unwrap_model(loss_module).should_discriminator_be_trained(global_step):
                    discriminator_loss, loss_dict_discriminator = loss_module(
                        images,
                        reconstructed_images,
                        extra_results_dict,
                        global_step=global_step,
                        last_layer=accelerator.unwrap_model(model).get_last_layer(),
                        mode="disc",
                    )

                    # Gather the losses across all processes for logging (if we use distributed training).
                    for k, v in loss_dict_discriminator.items():
                        if k in ["logits_real", "logits_fake", "extra_logits_fake"]:
                            if type(v) == torch.Tensor:
                                discriminator_logs["train/" + k] = v.cpu().item()
                            else:
                                discriminator_logs["train/" + k] = v
                        else:
                            discriminator_logs["train/" + k] = accelerator.gather(v).mean().item()

                    accelerator.backward(discriminator_loss)

                    if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(loss_module.parameters(), config.training.max_grad_norm)

                    discriminator_optimizer.step()
                    discriminator_lr_scheduler.step()

                    # log gradient norm before zeroing it
                    if (
                        accelerator.sync_gradients
                        and (global_step + 1) % config.experiment.log_grad_norm_every == 0
                        and accelerator.is_main_process
                    ):
                        log_grad_norm(loss_module, accelerator, global_step + 1)

                    discriminator_optimizer.zero_grad(set_to_none=True)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if config.training.use_ema:
                    ema_model.step(model.parameters())

                # wait for both generator and discriminator to settle
                batch_time_m.update(time.time() - end)
                end = time.time()

                if (global_step + 1) % config.experiment.log_every == 0:
                    samples_per_second_per_gpu = (
                        config.training.gradient_accumulation_steps * config.training.per_gpu_batch_size / batch_time_m.val
                    )
                    logger.info(
                        #f"Data (t): {data_time_m.val:0.4f}, {samples_per_second_per_gpu:0.2f}/s/gpu "
                        f"Batch (t): {batch_time_m.val:0.4f} "
                        f"LR: {lr_scheduler.get_last_lr()[0]:0.6f} "
                        f"Step: {global_step + 1} "
                        f"Total Loss: {autoencoder_logs['train/total_loss']:0.4f} "
                        f"Entropy Loss: {autoencoder_logs['train/entropy_loss']:0.4f} "
                        f"Sample Entropy: {autoencoder_logs['train/per_sample_entropy']:0.4f} "
                        f"Avg Entropy: {autoencoder_logs['train/avg_entropy']:0.4f} "
                        # f"Gen GAN Loss: {autoencoder_logs['train/gan_loss']:0.4f} "
                        f"Discr Loss: {discriminator_logs['train/discriminator_loss']:0.4f} "
                    )
                    logs = {
                        "lr": lr_scheduler.get_last_lr()[0],
                        "lr/generator": lr_scheduler.get_last_lr()[0],
                        "lr/discriminator": discriminator_lr_scheduler.get_last_lr()[0],
                        "samples/sec/gpu": samples_per_second_per_gpu,
                        "time/data_time": data_time_m.val,
                        "time/batch_time": batch_time_m.val,
                    }
                    logs.update(autoencoder_logs)
                    logs.update(discriminator_logs)
                    accelerator.log(logs, step=global_step + 1)

                    # resetting batch / data time meters per log window
                    batch_time_m.reset()
                    data_time_m.reset()

                # Save model checkpoint
                if (global_step + 1) % config.experiment.save_every == 0:
                    save_checkpoint(model, output_dir, accelerator, global_step + 1)

                    # Wait for everyone to save their checkpoint
                    accelerator.wait_for_everyone()

                # Generate images
                if (global_step + 1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    # Store the model parameters temporarily and load the EMA parameters to perform inference.
                    if config.training.get("use_ema", False):
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                    
                    generate_images(
                        model,
                        images[:config.training.num_generated_images],
                        fnames[:config.training.num_generated_images],
                        accelerator,
                        global_step + 1,
                        output_dir,
                        config.experiment.logger
                    )

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())

                # Evaluate reconstruction
                if (global_step + 1) % config.experiment.eval_every == 0:
                    # or all val images.
                    logger.info(f"Computing metrics on the validation set.")

                    if config.training.get("use_ema", False):
                        ema_model.store(model.parameters())
                        ema_model.copy_to(model.parameters())
                
                    eval_scores = eval_reconstruction(
                        model,
                        eval_dataloader,
                        accelerator,
                        evaluator
                    )

                    logger.info(
                        f"EVALUATION Step: {global_step + 1} "
                    )
                    logger.info(pprint.pformat(eval_scores))
                    if accelerator.is_main_process:
                        eval_log = {'eval/'+k: v for k, v in eval_scores.items()}
                        accelerator.log(eval_log, step=global_step + 1)

                    if config.training.get("use_ema", False):
                        # Switch back to the original model parameters for training.
                        ema_model.restore(model.parameters())
                    accelerator.wait_for_everyone()

                global_step += 1

            # Stop training if max steps is reached
            if global_step >= config.training.max_train_steps:
                accelerator.print(
                    f"Finishing training: Global step is >= Max train steps: {global_step} >= {config.training.max_train_steps}"
                )
                break
        # End for

    accelerator.wait_for_everyone()

    # Save checkpoint at the end of training
    save_checkpoint(model, output_dir, accelerator, global_step)

    accelerator.end_training()


@torch.no_grad()
def eval_reconstruction(
    model,
    eval_loader,
    accelerator,
    evaluator
):
    model.eval()
    evaluator.reset_metrics()
    local_model = accelerator.unwrap_model(model)

    for batch in eval_loader:
        images = batch["image"].to(
            accelerator.device, memory_format=torch.contiguous_format, non_blocking=True
        )
        original_images = torch.clone(images)
        reconstructed_images, model_dict = local_model(images)
        reconstructed_images = torch.clamp(reconstructed_images, 0.0, 1.0)
        original_images = torch.clamp(original_images, 0.0, 1.0)

        evaluator.update(original_images, reconstructed_images, model_dict["min_encoding_indices"])

    model.train()
    return evaluator.result()


@torch.no_grad()
def generate_images(model, original_images, fnames, accelerator, global_step, output_dir, logger_type):
    logger = get_logger(name="VQGAN", log_level="INFO")
    logger.info("Generating images...")
    original_images = torch.clone(original_images)
    # Generate images
    model.eval()
    dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        dtype = torch.bfloat16

    with torch.autocast("cuda", dtype=dtype, enabled=accelerator.mixed_precision != "no"):
        enc_tokens, _  = accelerator.unwrap_model(model).encode(original_images)

    reconstructed_images = accelerator.unwrap_model(model).decode(enc_tokens)
    model.train()
    
    images_for_wandb, images_for_tensorboard = make_viz_from_samples(
        original_images,
        reconstructed_images
    )
    
    if logger_type == "wandb":
        accelerator.get_tracker("wandb").log_images(
            {"Train Reconstruction": images_for_wandb}, step=global_step
        )
    else:
        accelerator.get_tracker("tensorboard").log_images(
            {"Train Reconstruction": images_for_tensorboard}, step=global_step
        )

    # Log locally
    root = Path(output_dir) / "train_images"
    os.makedirs(root, exist_ok=True)
    for i,img in enumerate(images_for_wandb):
        filename = f"{global_step:08}_s-{i:03}-{fnames[i]}.png"
        path = os.path.join(root, filename)
        img.save(path)


def save_checkpoint(model, output_dir, accelerator, global_step) -> Path:
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    logger = get_logger(name="VQGAN", log_level="INFO")

    # retrieve the model on all processes for deepspeed stage 3 to work then save on one process (we are not using stage 3 yet)
    state_dict = accelerator.get_state_dict(model)

    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "unwrapped_model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")

    accelerator.save_state(save_path)
    return save_path


def load_checkpoint(checkpoint_path: Path, accelerator):
    logger = get_logger(name="VQGAN", log_level="INFO")
    logger.info(f"Load checkpoint from {checkpoint_path}")

    accelerator.load_state(checkpoint_path)
    
    global_step = 0
    if os.path.exists(checkpoint_path / "metadata.json"):
        with open(checkpoint_path / "metadata.json", "r") as f:
            global_step = int(json.load(f)["global_step"])

        logger.info(f"Resuming at global_step {global_step}")
    return global_step


def log_grad_norm(model, accelerator, global_step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().data
            grad_norm = (grads.norm(p=2) / grads.numel()).item()
            accelerator.log({"grad_norm/" + name: grad_norm}, step=global_step)


if __name__ == "__main__":
    main()