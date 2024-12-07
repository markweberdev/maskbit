"""This file contains the definition of the data loader using webdataset.

We thank the following public implementations for inspiring this code:
    https://github.com/mlfoundations/open_clip/blob/main/src/training/data.py
    https://github.com/huggingface/open-muse/blob/main/training/data.py
"""

import math
from typing import List, Union, Text

import webdataset as wds
from torch.utils.data import default_collate
from torchvision import transforms


def filter_keys(key_set):
    def _f(dictionary):
        return {k: v for k, v in dictionary.items() if k in key_set}

    return _f


class ImageNetTransform:
    def __init__(
        self,
        resolution,
        use_aspect_ratio_aug: bool = True,
        use_random_crop: bool = True,
        min_scale: float = 0.05,
        interpolation: Text = "bilinear"
    ):
        """ Initialize the WebDatasetReader object.

        Args:
            resolution -> int: The resolution of the images.
            use_aspect_ratio_aug -> bool: Flag indicating whether to change the aspect ratio for augmentation. Defaults to `True`.
            use_random_crop -> bool: Flag indicating whether to use random cropping. Defaults to `True`.
            min_scale -> float: The minimum scale when applyign random resizing. Defaults to 0.05.
            interpolation -> Text: The interpolation technique for resizing. Must be `bilinear`, `bicubic` or `nearest`.
        """
        self.interpolation_lookup = {
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "nearest": transforms.InterpolationMode.NEAREST,
            "bicubic": transforms.InterpolationMode.BICUBIC,
        }
        if use_aspect_ratio_aug:
            min_aspect_ratio = 3.0/4.0
            max_aspect_ratio = 4.0/3.0
        else:
            min_aspect_ratio = 1.0
            max_aspect_ratio = 1.0

        if use_random_crop:
            self.train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(
                        resolution,
                        scale=(min_scale, 1.0),
                        ratio=(min_aspect_ratio, max_aspect_ratio),
                        interpolation=self.interpolation_lookup[interpolation]
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            if min_scale != 1.0:
                raise ValueError("min_scale must be 1.0 when use_random_crop is False")
            if min_aspect_ratio != 1.0 or max_aspect_ratio != 1.0:
                raise ValueError("min_aspect_ratio and max_aspect_ratio must be 1.0 when use_random_crop is False")
            
            self.train_transform = transforms.Compose(
                [
                    transforms.Resize(resolution, interpolation=self.interpolation_lookup[interpolation]),
                    transforms.CenterCrop(resolution),
                    transforms.ToTensor(),
                ]
            )
        self.eval_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=self.interpolation_lookup[interpolation]),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
            ]
        )


class SimpleImagenet:
    def __init__(
        self,
        train_shards_path: Union[Text, List[Text]],
        eval_shards_path: Union[Text, List[Text]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers_per_gpu: int,
        resolution: int = 256,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        use_aspect_ratio_aug: bool = True,
        use_random_crop: bool = True,
        min_scale: float = 0.05,
        interpolation: Text = "bilinear",
    ):
        """
        Initialize a WebDatasetReader for ImageNet.

        Args:
            train_shards_path -> Union[Text, List[Text]]: Path of the training shards.
            eval_shards_path -> Union[Text, List[Text]]: Path of the evaluation shards.
            num_train_examples -> int: Number of training examples.
            per_gpu_batch_size -> int: Batch size per GPU.
            global_batch_size -> int: Global batch size.
            num_workers_per_gpu -> int: Number of workers per GPU.
            resolution -> int: Resolution of the images. Defaults to 256.
            shuffle_buffer_size -> int: Buffer size for shuffling. Defaults to 1000.
            pin_memory -> bool: Whether to pin memory. Defaults to False.
            persistent_workers -> bool: Whether to use persistent workers. Defaults to False.
            use_aspect_ratio_aug -> bool: Whether to use aspect ratio augmentation. Defaults to True.
            use_random_crop -> bool: Whether to use random crop. Defaults to True.
            min_scale -> float: Minimum scale for random crop. Defaults to 0.05.
            interpolation -> Text: Interpolation method. Defaults to "bilinear".
        """
        transform = ImageNetTransform(
            resolution,
            use_aspect_ratio_aug=use_aspect_ratio_aug,
            use_random_crop=use_random_crop,
            min_scale=min_scale,
            interpolation=interpolation
            )

        train_processing_pipeline = [
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
            wds.rename(
                image="jpg;png;jpeg;webp",
                class_id="cls",
                handler=wds.warn_and_continue,
                ),
            wds.map(filter_keys(set(["image", "class_id", "filename"]))),
            wds.map_dict(
                image=transform.train_transform,
                class_id=lambda x: int(x),
                handler=wds.warn_and_continue,
            ),
        ]

        test_processing_pipeline = [
            wds.decode(wds.autodecode.ImageHandler("pil", extensions=["webp", "png", "jpg", "jpeg"])),
            wds.rename(
                image="jpg;png;jpeg;webp",
                class_id="cls",
                # filename="__key__",
                handler=wds.warn_and_continue,
                ),
            wds.map(filter_keys(set(["image", "class_id", "filename"]))),
            wds.map_dict(
                image=transform.eval_transform,
                class_id=lambda x: int(x),
                handler=wds.warn_and_continue,
            ),
        ]


        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path),
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.shuffle(shuffle_buffer_size),
            *train_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=False, collation_fn=default_collate),
        ]

        num_batches = math.ceil(num_train_examples / global_batch_size)
        num_worker_batches = math.ceil(num_train_examples / (global_batch_size * num_workers_per_gpu))  # per dataloader worker
        num_batches = num_worker_batches * num_workers_per_gpu
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

        # Create eval dataset and loader
        pipeline = [
            wds.SimpleShardList(eval_shards_path),
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=wds.ignore_and_continue),
            *test_processing_pipeline,
            wds.batched(per_gpu_batch_size, partial=True, collation_fn=default_collate),
        ]
        self._eval_dataset = wds.DataPipeline(*pipeline)
        self._eval_dataloader = wds.WebLoader(
            self._eval_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers_per_gpu,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @property
    def eval_dataloader(self):
        return self._eval_dataloader


if __name__ == "__main__":
    import tqdm
    imagenet_train_path = "/PATH/imagenet_shards/train/imagenet-train-{0000..0252}.tar"
    imagenet_test_path = "/PATH/imagenet_shards/val/imagenet-val-{0000..0009}.tar"
    imagenet = SimpleImagenet(
        imagenet_train_path,
        imagenet_test_path,
        num_train_examples=500_000,
        per_gpu_batch_size=16,
        global_batch_size=16*4,
        num_workers_per_gpu=4,
        resolution=256,
        shuffle_buffer_size=1000,
        pin_memory=True,
        persistent_workers=False,
        use_aspect_ratio_aug=True,
        interpolation="bilinear"
    )
    imagenet_train_dataloader = imagenet.train_dataloader
    imagenet_test_dataloader = imagenet.eval_dataloader

    iter_loader = iter(imagenet_test_dataloader)
    key_set = set()
    for batch in tqdm.tqdm(iter_loader):
        key_set |= set(batch["__key__"])
    
    print(len(key_set))