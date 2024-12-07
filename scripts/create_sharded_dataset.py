"""This file contains a script for creating a webdataset."""

import sys
import os
import random
import argparse
import glob

from collections import OrderedDict

import tqdm

import webdataset as wds

from data.imagenet_classes import IMAGENET2012_CLASSES


parser = argparse.ArgumentParser("""Generate sharded dataset from original ImageNet data.""")
parser.add_argument("--splits", default="train,val", help="which splits to write")
parser.add_argument(
    "--filekey", action="store_true", help="use file as key (default: index)"
)
parser.add_argument("--maxsize", type=float, default=1e9)
parser.add_argument("--maxcount", type=float, default=5079)
parser.add_argument(
    "--shards", default="./shards", help="directory where shards are written"
)
parser.add_argument(
    "--data",
    default="./data",
    help="directory containing ImageNet data distribution suitable for torchvision.datasets",
)
args = parser.parse_args()


assert args.maxsize > 10000000
assert args.maxcount < 1000000


if not os.path.isdir(os.path.join(args.data, "train")):
    print(f"{args.data}: should be directory containing ImageNet", file=sys.stderr)
    sys.exit(1)


if not os.path.isdir(os.path.join(args.shards, ".")):
    print(f"{args.shards}: should be a writable destination directory for shards", file=sys.stderr)
    sys.exit(1)


splits = args.splits.split(",")


def readfile(fname):
    "Read a binary file from disk."
    with open(fname, "rb") as stream:
        return stream.read()


all_keys = set()


def write_dataset(imagenet, base="./shards", split="train"):

    image_paths = glob.glob(os.path.join(imagenet, split, "**/*.JPEG"))

    nimages = len(image_paths)
    print("# nimages", nimages)

    # We shuffle the indexes to make sure that we
    # don't get any large sequences of a single class
    # in the dataset.
    indexes = list(range(nimages))
    random.shuffle(indexes)

    # This is the output pattern under which we write shards.
    pattern = os.path.join(base, split, f"imagenet-{split}-%04d.tar")
    os.makedirs(os.path.join(base, split), exist_ok=True)

    fname_to_labels = OrderedDict()
    for i, (k,v) in enumerate(IMAGENET2012_CLASSES.items()):
        fname_to_labels[k] = (i, v)

    # print(len(fname_to_labels))

    with wds.ShardWriter(pattern, maxsize=int(args.maxsize), maxcount=int(args.maxcount)) as sink:
        for i in tqdm.tqdm(indexes):

            img_path = image_paths[i]

            # Internal information from the ImageNet dataset
            # instance: the file name and the numerical class.
            # cls_id, human_label = fname_to_labels[os.path.splitext(os.path.basename(img_path))[0].split('_')[0]]
            cls_id, human_label = fname_to_labels[img_path.split('/')[-2]]

            # Read the JPEG-compressed image file contents.
            image = readfile(img_path)

            # Construct a unique key from the filename.
            key = os.path.splitext(os.path.basename(img_path))[0]

            # Useful check.
            assert key not in all_keys
            all_keys.add(key)

            # Construct a sample.
            xkey = key if args.filekey else "%07d" % i
            sample = {"__key__": xkey, "jpg": image, "cls": cls_id}

            # Write the sample to the sharded tar archives.
            sink.write(sample)


for split in splits:
    print("# split", split)
    write_dataset(args.data, base=args.shards, split=split)