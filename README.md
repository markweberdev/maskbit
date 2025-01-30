# MaskBit: Embedding-free Image Generation via Bit Tokens

<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/maskbit-embedding-free-image-generation-via/image-generation-on-imagenet-256x256)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?p=maskbit-embedding-free-image-generation-via)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://weber-mark.github.io/projects/maskbit.html)&nbsp;&nbsp;
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.16211)&nbsp;&nbsp;

</div>

This repository contains an implementation of the paper "MaskBit: Embedding-free Image Generation via Bit Tokens" accepted to TMLR with **featured and reproducibility certifications**.

We present a modernized VQGAN+ and a novel image generation framework leveraging bit tokens. As a result, MaskBit uses a shared representation in the tokenizer and generator, which yields state-of-the-art results (at time of publication) while using a significantly smaller model compared to autoregressive models. 


<p>
<img src="assets/teaser_image.png" alt="teaser" width=90% height=90%>
</p>
<p>
<img src="assets/arch_maskbit.png" alt="teaser" width=90% height=90%>
</p>

## 🚀 Contributions

#### We study the key ingredients of recent closed-source VQGAN tokenizers and develop a publicly available, reproducible, and high-performing VQGAN model, called VQGAN+, achieving a significant improvement of 6.28 rFID over the original VQGAN developed three years ago. 

#### Building on our improved tokenizer framework, we leverage modern Lookup-Free Quantization (LFQ). We analyze the latent representation and observe that embedding-free bit token representation exhibits highly structured semantics. 

#### Motivated by these discoveries, we develop a novel embedding-free generation framework, MaskBit, which builds on top of the bit tokens and achieves state-of-the-art performance on the ImageNet 256×256 class-conditional image generation benchmark. 


## Updates
- 12/06/2024: Code release and tokenizer models. 
- 12/01/2024: Accepted to TMLR with **featured and reproducibility certifications**. 
- 09/24/2024: The [tech report](https://arxiv.org/abs/2409.16211) of MaskBit is available.


## Model Zoo

All models are trained on ImageNet with an input shape of 256x256. All models downsample the images to a spatial size of 16x16, leading to a latent representation of 16x16xK bits per image.

### Tokenizer

| Model | Link | reconstruction FID | config | 
| ------------- | ------------- | ------------- | ------------- |
| VQGAN+ (10 bits, from the paper) | [checkpoint](https://huggingface.co/markweber/vqgan_plus_paper)| 1.67 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/vqgan_plus_10bit.yaml) |
| VQGAN+ (10 bits) | TODO | 1.52 |  |
| VQGAN+ (12 bits) | [checkpoint](https://huggingface.co/markweber/vqgan_plus_12bit) | 1.39 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/vqgan_plus_12bit.yaml) |
| ------------- | ------------- | ------------- | ------------- |
| MaskBit-Tokenizer (10 bits) | [checkpoint](https://huggingface.co/markweber/maskbit_tokenizer_10bit) | 1.76 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/maskbit_tokenizer_10bit.yaml) |
| MaskBit-Tokenizer (12 bits) | [checkpoint](https://huggingface.co/markweber/maskbit_tokenizer_12bit) | 1.52 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/maskbit_tokenizer_12bit.yaml) |
| MaskBit-Tokenizer (14 bits) | [checkpoint](https://huggingface.co/markweber/maskbit_tokenizer_14bit) | 1.37 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/maskbit_tokenizer_14bit.yaml) |
| MaskBit-Tokenizer (16 bits) | [checkpoint](https://huggingface.co/markweber/maskbit_tokenizer_16bit) | 1.29 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/maskbit_tokenizer_16bit.yaml) |
| MaskBit-Tokenizer (18* bits) | [checkpoint](https://huggingface.co/markweber/maskbit_tokenizer_18bit) | 1.16 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/tokenizer/maskbit_tokenizer_18bit.yaml) |
| ------------- | ------------- | ------------- | ------------- |
| Taming-VQGAN (10 bits) | [checkpoint](https://huggingface.co/markweber/taming_vqgan) | 7.96 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/external/taming_vqgan_tokenizer.yaml) |
| MaskGIT-Tokenizer (10 bits) | [checkpoint](https://huggingface.co/markweber/maskgit_tokenizer) | 1.96 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/external/maskgit_tokenizer.yaml) |

*In practice only 17 bits are used, as one bit does not change. We did not put any effort into fixing "dead" bits, as such large vocabulary was not needed for ImageNet.

Since the initial release of the paper, we have made some small changes and to the training recipe to improve the reconstruction quality of the tokenizer. 

Please note that these models are trained only on limited academic dataset ImageNet, and they are only for research purposes.

### Generator

| Model | Link | generation FID | config | 
| ------------- | ------------- | ------------- | ------------- |
| MaskBit-Generator (10 bits), 64 steps | Coming soon | 1.68 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/generator/maskbit_generator_10bit.yaml) |
| MaskBit-Generator (12 bits), 64 steps | Coming soon | 1.65 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/generator/maskbit_generator_12bit.yaml) |
| MaskBit-Generator (14 bits), 64 steps | Coming soon | 1.62 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/generator/maskbit_generator_14bit.yaml) |
| MaskBit-Generator (16 bits), 64 steps | Coming soon | 1.64 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/generator/maskbit_generator_16bit.yaml) |
| MaskBit-Generator (18 bits), 64 steps | Coming soon | 1.67 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/generator/maskbit_generator_18bit.yaml) |
| ------------- | ------------- | ------------- | ------------- |
| MaskBit-Generator (14 bits), 128 steps | Coming soon | 1.56 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/generator/maskbit_generator_14bit_128steps.yaml) |
| MaskBit-Generator (14 bits), 256 steps | Coming soon | 1.52 | [config](https://github.com/markweberdev/maskbit/blob/main/configs/generator/maskbit_generator_14bit_256steps.yaml)  |

Test-time hyper-parameters such as `randomize_temperature`, `guidance_scale`, `scale_pow`, and `num_steps` can be tuned after training on any model. The optimal choice for these hyper-parameters can vary per model configuration. `num_steps` is the main hyper-parameter to control the speed-performance trade-off during training and inference. 

Please note that these models are trained only on limited academic dataset ImageNet, and they are only for research purposes. We will release the Stage-II models soon.

## Installation

The codebase was tested with Python 3.9 and Pytorch 2.2.2. After setting up pytorch, you can use the following script to install additional requirements.

```shell
pip3 install -r requirements.txt
```

## Training

### Tokenizer (Stage-I)

Please first follow the install guide and the [data preparation doc](https://github.com/markweberdev/maskbit/blob/main/docs/prepare_data.md). 

We use the accelerate library for multi-device training, which means the following command needs to be started on each worker:

```python3
PYTHONPATH=./ WORKSPACE=./ accelerate launch --num_machines=1  --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_tokenizer.py config=./configs/tokenizer/maskbit_tokenizer_10bit.yaml
```

For more instructions on how to use the accelerate library, we refer to their [website](https://huggingface.co/docs/accelerate/v1.2.0/en/index). Moreover, run specific config changes can also be done by passing the config changes on the command line. For example ```training.per_gpu_batch_size=32``` would use a batchsize of 32 for this run.

### Generator (Stage-II)

Please first follow the install guide and the [data preparation doc](https://github.com/markweberdev/maskbit/blob/main/docs/prepare_data.md). 

We use the accelerate library for multi-device training, which means the following command needs to be started on each worker:

```python3
PYTHONPATH=./ WORKSPACE=./ accelerate launch --num_machines=1  --machine_rank=0 --main_process_ip=127.0.0.1 --main_process_port=9999 --same_network scripts/train_maskbit.py config=./configs/generator/maskbit_generator_10bit.yaml
```

For more instructions on how to use the accelerate library, we refer to their [website](https://huggingface.co/docs/accelerate/v1.2.0/en/index). Moreover, run specific config changes can also be done by passing the config changes on the command line. For example ```training.per_gpu_batch_size=32``` would use a batchsize of 32 for this run.

We will release checkpoints and configs for the generator soon.

## Testing on ImageNet-1K Benchmark

### Tokenizer (Stage-I)

Please first follow the install guide and the [data preparation doc](https://github.com/markweberdev/maskbit/blob/main/docs/prepare_data.md). 

After choosing the model config and checkpoint, the following command will run the evaluation:

```python3
PYTHONPATH=./ python3 scripts/eval_tokenizer.py config=./configs/tokenizer/maskbit_tokenizer_12bit.yaml experiment.vqgan_checkpoint=/PATH_TO_MODEL/maskbit_tokenizer_12bit.bin
```

### Generator (Stage-II)

Coming soon.

## Detailed Results

| Model | reconstruction FID | Inception Score | PSNR | SSIM | Codebook Usage |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| VQGAN+ (10 bits, from the paper) | 1.67 | 186.5 | 20.9 | 0.53 | 1.0 |
| VQGAN+ (10 bits) | 1.52 | 182.4 | 21.1 | 0.54 | 1.0 |
| VQGAN+ (12 bits) | 1.39 | 193.9 | 21.0 | 0.55 | 1.0 |
| ------------- | ------- | ------- | ------- | ------- | ------- |
| MaskBit-Tokenizer (10 bits) | 1.76 | 177.6 | 20.8 | 0.53 | 1.0 |
| MaskBit-Tokenizer (12 bits) | 1.52 | 184.3 | 21.2 | 0.55 | 1.0 |
| MaskBit-Tokenizer (14 bits) | 1.37 | 190.3 | 21.5 | 0.56 | 1.0 |
| MaskBit-Tokenizer (16 bits) | 1.29 | 193.6 | 21.8 | 0.58 | 1.0 |
| MaskBit-Tokenizer (18* bits) | 1.16 | 197.8 | 22.0 | 0.59 | 0.5 |
| ------------- | ------- | ------- | ------- | ------- | ------- |
| Taming-VQGAN (10 bits) | 7.96 | 115.9 | 20.18 | 0.52 | 1.0 |
| MaskGIT-Tokenizer (10 bits) | 1.96 | 178.3 | 18.6 | 0.47 | 0.45 |

## Citing
If you use our work in your research, please use the following BibTeX entry.

```BibTeX
@article{weber2024maskbit,
  author    = {Mark Weber and Lijun Yu and Qihang Yu and Xueqing Deng and Xiaohui Shen and Daniel Cremers and Liang-Chieh Chen},
  title     = {MaskBit: Embedding-free Image Generation via Bit Tokens},
  journal   = {arXiv:2409.16211},
  year      = {2024}
}
```
