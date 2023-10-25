# Model2Dataset 


The goal of this repo is to compute embeddings or generated samples on a dataset using pretrained models and save
them into disk into a synthetic.

# Features

- Support for Timm to compute features or classes
- Support for HuggingFace models to compute features or classes
- Support for LLAVA (captioning)

# How to install?

```bash
git clone https://github.com/mehdidc/model2dataset
cd model2dataset
python setup.py develop
```

# How to use

example with LLAVA:

Write follow into `config.yaml`
```yaml
dataset:
  root: /path/0000000.tar
  type: webdataset
  entries:
    image: png;jpg;jpeg;webp
  inputs:
    - image
    - llava_image
  batch_size: 4
  workers: 4
output:
  per_shard: 4
  folder: out
  outputs:
    - image.jpg
    - llava_caption.txt
pipeline:
  llava:
    type: llava
    image: llava_image
    model_path: /path/llava-v1.5-13b
    model_base: null
    query: "Can you describe this image?"
    conv_mode: null
    temperature: 0.2
    max_new_tokens: 1024
```

then

`model2dataset --config config.yaml`