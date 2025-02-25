import os
import argparse
import math
from pathlib import Path
import pprint

import torch
from omegaconf import OmegaConf
import tqdm

from modeling.conv_vqgan import ConvVQModel
from modeling.bert import Bert, LFQBert
from modeling.modules import sample
from utils.adm_eval_suite import Evaluator

import tensorflow.compat.v1 as tf

TRAIN_SET_STATISTICS_256 = "train_imagenet256_stats.npz"
TRAIN_SET_STATISTICS_512 = "train_imagenet512_stats.npz"




@torch.no_grad()
def get_tokenizer(config, tokenizer_path):
    tokenizer_model = ConvVQModel(config.model.vq_model, legacy=False)
    tokenizer_model.load_pretrained(tokenizer_path)
    tokenizer_model.eval()
    tokenizer_model.requires_grad_(False)
    return tokenizer_model


@torch.no_grad()
def get_generator(config, generator_path):
    stage2_model_cls = {
        "bert": Bert,
        "lfq_bert": LFQBert,
    }[config.model.mlm_model.model_cls]
    
    generator_model = stage2_model_cls(
        img_size=config.dataset.preprocessing.resolution,
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


def main(
    config_file,
    batchsize: int = 100,
    res: int = 256,
    tokenizer_path: str = "",
    generator_path: str = "",
    device: str = "cuda:0",
):
    config = OmegaConf.load(config_file)

    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    if config.model.vq_model.quantizer_type == "lookup-free":
        num_codebook_entries = 2 ** config.model.vq_model.token_size
        config.model.vq_model.codebook_size = num_codebook_entries
        config.model.mlm_model.mask_token = int(2 ** (math.log2(num_codebook_entries) // config.model.mlm_model.codebook_splits))
    else:
        num_codebook_entries = config.model.vq_model.codebook_size
        config.model.mlm_model.mask_token = int(2 ** (math.log2(num_codebook_entries) // config.model.mlm_model.codebook_splits))

    tokenizer_model = get_tokenizer(config, tokenizer_path).to(device)
    generator_model = get_generator(config, generator_path).to(device)

    ##################################
    # EVALUATION STUFF.              #
    ##################################
    with torch.no_grad():

        tokenizer_model.eval()
        generator_model.eval()
        total_samples = 50_000

        generated_list = []

        tf_config = tf.ConfigProto(
            allow_soft_placement=True  # allows DecodeJpeg to run on CPU in Inception graph
        )
        tf_config.gpu_options.allow_growth = True
        evaluator = Evaluator(tf.Session(config=tf_config))

        print("warming up TensorFlow...")
        evaluator.warmup()

        # This is important due to how the Inception Score is computed in the tensorflow suite.
        # The computation is done batchwise, hence we need to shuffle everithing to have a good representation per batch.
        labels = torch.randperm(1000, dtype=torch.int, device=device)
        labels = labels.repeat(50)

        print("Running generation...")
        for i in tqdm.tqdm(range(total_samples//batchsize), desc="Generating samples", position=0):
            y = labels[batchsize*i: batchsize*(i+1)].long()

            generated_samples, _ = sample(
                generator_model,
                tokenizer_model,
                num_samples=batchsize,
                labels=y,
                softmax_temperature=1.0,
                randomize_temperature=config.model.mlm_model.randomize_temperature,
                mask_schedule_strategy=config.model.mlm_model.gen_mask_schedule_strategy,
                num_steps=config.model.mlm_model.num_steps,
                guidance_scale=config.model.mlm_model.guidance_scale,
                mask_token=config.model.mlm_model.mask_token,
                patch_size=int(res // 2**(config.model.vq_model.num_resolutions - 1)),
                guidance_annealing=config.model.mlm_model.guidance_annealing,
                use_sampling_annealing=config.model.mlm_model.use_sampling_annealing,
                scale_pow=config.model.mlm_model.scale_pow,
                codebook_size=config.model.vq_model.codebook_size,
                codebook_splits=config.model.mlm_model.codebook_splits,
                use_tqdm=True,
            )
                
            generated_samples = torch.clamp(generated_samples, 0.0, 1.0)
            generated_samples = (generated_samples * 255.0).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            generated_list.append(generated_samples)

        if res == 256:
            stat_file = TRAIN_SET_STATISTICS_256
        elif res == 512:
            stat_file = TRAIN_SET_STATISTICS_512
        else:
            raise ValueError("res must be 256 or 512")
        
        current_file_dir = Path(__file__).parent

        stats_file_path = (
            current_file_dir 
            / Path("..") 
            / "metrics" 
            / "stats" 
            / stat_file
        ).resolve()

        print("Running evaluation...")

        ref_stats = evaluator.read_statistics(stats_file_path, None)

        sample_acts = evaluator.compute_activations(generated_list)
        sample_stats = evaluator.compute_statistics(sample_acts)

        eval_scores = {
            "InceptionScore": evaluator.compute_inception_score(sample_acts),
            "FID": sample_stats.frechet_distance(ref_stats),
        }

    print("EVALUATION")
    print(f"Results for {config.model.vq_model.token_size} bits with {config.model.mlm_model.num_steps} steps.")
    pprint.pprint(eval_scores)
    return eval_scores



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Eval script")
    parser.add_argument('--config', type=str, help='Path to the config file')
    parser.add_argument('--batchsize', type=int, default=100, help='Batchsize (int).')
    parser.add_argument('--res', type=int, default=256, help='Resolution (int).')
    parser.add_argument('--tokenizer', type=str, help='Path to the tokenizer file')
    parser.add_argument('--generator', type=str, help='Path to the generator file')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device')

    args = parser.parse_args()

    main(
        config_file=args.config,
        batchsize=args.batchsize,
        res=args.res,
        tokenizer_path=args.tokenizer,
        generator_path=args.generator,
        device=args.device
    )
