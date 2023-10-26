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
  root: /path/{0000000..0139827}.tar
  type: webdataset
  entries:
    image: png;jpg;jpeg;webp
    caption: txt
  inputs:
    - image
    - llava_transformed_image
    - caption
  batch_size: 16
  workers: 8

pipeline:

  image_to_caption:
    type: llava
    image_transform:
      image: llava_transformed_image
    image: llava_transformed_image
    model_path: /path/llava-v1.5-7b
    model_base: null
    prompt: "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>
Can you describe the image in few words ? ASSISTANT:"
    query: null
    conv_mode: null
    temperature: 0.2
    max_new_tokens: 1
    load_4bit: False
    load_8bit: False
    output: llava_caption
  
output:
  per_shard: 10000
  folder: out
  outputs:
    image: jpg
    llava_caption: txt
```

then

`model2dataset --config config.yaml`