## Data Preparation

We use the [webdataset](https://github.com/webdataset/webdataset) format for data loading. Before running any evaluation or training any data needs to be converted into webdataset format. An example script to convert ImageNet to wds format is provided [here](../scripts/create_sharded_dataset.py).

```python3
PYTHONPATH=./ python3 scripts/create_sharded_dataset.py --shards=/OUTPUT_FOLDER/ --data=/PATH_TO_IMAGENET/
```
