"""
Things to support:

    - image captioners:
        - LLAVA
        - OpenFlamingo
        - OBELICS
        - CogVLM
    - OCR
        - nougat
    - Segmentation
        - Segment Everything
    - image generators
        - SDXL
    - LLMs 
        - Mistral
        - Phi
        - T5, UMT5, etc
    - SEED, VAE
"""
from omegaconf import OmegaConf
import PIL.Image
import torch
try:
    import timm
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    has_timm = True
except ImportError:
    has_timm = False

try:
    import transformers
    from transformers import AutoFeatureExtractor
    has_huggingface = True
except ImportError:
    has_huggingface = False

from llava_model import LLAVACaptioner

class TimmFeatureExtractor:

    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        self.model = timm.create_model(cfg.model, pretrained=True).to(self.device)
        config = resolve_data_config({}, model=self.model)
        self.image_transform = create_transform(**config)
    def predict(self, inputs):
        image = inputs[self.cfg.image].to(self.device)
        return self.model(image)


class HFFeatureExtractor:

    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        fe = AutoFeatureExtractor.from_pretrained(cfg.model)
        def prepro(image):
            data = fe(image, return_tensors="pt")
            return data['pixel_values'][0]
        self.image_transform = prepro
        self.model = getattr(transformers, cfg.model_class).from_pretrained(cfg.model).to(self.device)
    
    def predict(self, inputs):
        image = inputs[self.cfg.image].to(self.device)
        fe = self.model(image, **self.cfg.predict_options if self.cfg.predict_options else {})
        dump = {}
        for out in self.cfg.outputs:
            out_values = getattr(fe, out)
            dump[out] = out_values
        return dump

  
class Pipeline:

    def __init__(self, steps):
        self.steps = steps
    
    def image_transform(self, key, image):
        out = {}
        for step in self.steps:
            if hasattr(step, "image_transform"):
                out[step.name + "_" + key] = step.image_transform(image)
        return out
    
    def predict(self, inputs):
        for step in self.steps:
            outputs = step.predict(inputs)
            for k, v in outputs.items():
                inputs[step.name + "_" + k] = v
        return inputs

class PipelineImageTransform():

    def __init__(self, pipe):
        self.pipe = pipe
    
    def __call__(self, inputs):
        for data in inputs:
            data_new = {}
            data_new.update(data)
            for k, v in data.items():
                if type(v) == PIL.Image.Image:
                    data_out = self.pipe.image_transform(k, v)
                    data_new.update(data_out)
            yield data_new

def build_single(config):
    if config.type == "timm":
        return TimmFeatureExtractor(config)
    elif config.type == "huggingface":
        return HFFeatureExtractor(config)
    elif config.type == "llava":
        return LLAVACaptioner(config)
    else:
        raise ValueError(f"Unknown feature extractor type: {config.type}")


def build_pipeline(config):
    steps = []
    keys = list(config.keys())
    for k in keys:
        v = getattr(config, k)
        fe = build_single(v)
        fe.name = k
        steps.append(fe)
    return Pipeline(steps)

cfg_str = """
#fe1:
#    type: huggingface
#    image: image
#    model: microsoft/resnet-50
#    model_class: ResNetModel
#    predict_options:.
#    outputs: 
#        - last_hidden_state
fe2:
    type: llava
    image: fe2/image
    model_path: /p/project/laionize/cherti1/llava-v1.5-13b
    model_base: null
    query: "Can you describe this image?"
    conv_mode: null
    temperature: 0.2
    max_new_tokens: 1024
"""

if __name__ == "__main__":
    from PIL import Image
    import requests
    from PIL import Image
    from io import BytesIO
    def load_image(image_file):
        if image_file.startswith('http') or image_file.startswith('https'):
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_file).convert('RGB')
        return image
    cfg = OmegaConf.create(cfg_str)
    pipe = build(cfg)
    image = load_image("1200px-Stonehenge.jpg")
    inputs = pipe.image_transform(key="image", image=image)
    print(inputs.keys())
    out = pipe.predict(inputs)
    print(out)