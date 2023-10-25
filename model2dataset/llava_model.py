import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
    
#disable_torch_init()

class LLAVACaptioner:

    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cfg = cfg
        model_name = get_model_name_from_path(cfg.model_path)
        tokenizer, model, image_processor, context_len = load_pretrained_model(cfg.model_path, cfg.model_base, model_name)

        qs = cfg.query
        assert qs
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        if 'llama-2' in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"

        if cfg.conv_mode is not None and conv_mode != cfg.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, cfg.conv_mode, cfg.conv_mode))
        else:
            cfg.conv_mode = conv_mode

        conv = conv_templates[cfg.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        def transform(image):
            return image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half()[0]
        self.image_transform = transform
        self.prompt = prompt
        self.tokenizer = tokenizer
        self.conv = conv
        self.model = model.to(self.device)
    
    def predict(self, inputs):
        image_tensor = inputs[self.cfg.image].to(self.device)
        prompt = self.prompt
        tokenizer = self.tokenizer
        conv = self.conv
        nb = len(image_tensor)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).repeat(nb, 1).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        print(input_ids.shape, image_tensor.shape)
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=self.cfg.temperature,
                max_new_tokens=self.cfg.max_new_tokens,
                use_cache=True,
                #stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        outputs = [o.strip() for o in outputs]
        #outputs = [(o[:-len(stop_str)] if o.endswith(stop_str) else o) for o in outputs]
        outputs = [(o[0:o.index(stop_str)] if (stop_str in o) else o) for o in outputs]
        outputs = [o.strip() for o in outputs]
        return {"caption": outputs}