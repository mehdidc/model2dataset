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

class String:

    def __init__(self, cfg):
        self.cfg = cfg
    
    def predict(self, inputs):
        captions = {k:v for k, v in inputs.items() if type(v[0]) == str}
        values = []
        nb = len(captions[list(captions.keys())[0]])
        for i in range(nb):
            value = self.cfg.format.format(**{k: v[i] for k, v in captions.items()})
            values.append(value)
        return {self.cfg.output: values}
  
class Pipeline:

    def __init__(self, steps):
        self.steps = steps
    
    def image_transform(self, inputs):
        dump = {}
        for step in self.steps:
            if hasattr(step, "image_transform"):
                for k, v in step.cfg.image_transform.items():
                    dump[v] = step.image_transform(inputs[k])
        return dump
    
    def predict(self, inputs):
        for step in self.steps:
            outputs = step.predict(inputs)
            inputs.update(outputs)
        return inputs

class PipelineImageTransform():

    def __init__(self, pipe):
        self.pipe = pipe
    
    def __call__(self, inputs):
        for data in inputs:
            data_new = {}
            data_new.update(data)
            data_out = self.pipe.image_transform(data)
            data_new.update(data_out)
            yield data_new




def build_pipeline(config):
    steps = []
    keys = list(config.keys())
    for k in keys:
        v = getattr(config, k)
        fe = build_model(v)
        fe.name = k
        steps.append(fe)
    return Pipeline(steps)

def build_model(config):
    if config.type == "timm":
        return TimmFeatureExtractor(config)
    elif config.type == "huggingface":
        return HFFeatureExtractor(config)
    elif config.type == "llava":
        return LLAVACaptioner(config)
    elif config.type == "string":
        return String(config)
    else:
        raise ValueError(f"Unknown feature extractor type: {config.type}")
    

class Filters:

    def __init__(self, filters):
        self.filters = filters
    
    def filter(self, input):
        return all([f.filter(input) for f in self.filters])

class ExprFilter:

    def __init__(self, cfg):
        self.cfg = cfg
    
    def filter(self, inputs):
        locals().update(inputs)
        return eval(self.cfg.expr)

def build_filters(cfg):
    filters = []
    for k, v in cfg.items():
        filters.append(build_filter(v))
    return Filters(filters)

def build_filter(cfg):
    if cfg.type == "expr":
        return ExprFilter(cfg)
    else:
        raise ValueError(f"Unknown filter type: {v.type}")