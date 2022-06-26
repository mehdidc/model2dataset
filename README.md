# Compute Embeddings 


The goal of this repo is to compute embeddings  on dataset using pretrained models and save
them into disk.

# Features

- Support for Timm and HuggingFace libraries for pretrained models
- Support Webdataset and torchvision's ImageFolder for data loading
- Support for computing embeddings in distributed manner 

# How to install?

```bash
git clone https://github.com/mehdidc/compute_embeddings
cd compute_embeddings
python setup.py install
```

# How to use?

## Timm example

`compute_embeddings --dataset__type  image_folder --dataset_root root --batches_per_chunk 10  --library=timm --model "resnet50"  --batch_size 64 --distributed --out_folder out`

## HuggingFace example

`compute_embeddings --dataset__type  image_folder --dataset_root root --batches_per_chunk 10  --library=huggingface --model_class RegNetModel --model "facebook/regnet-y-040"  --batch_size 64 --distributed --out_folder out`
