#Adatapted from https://github.com/jaeseokbyun/MACIR/blob/main/models.py

from copy import copy
from os import PathLike
from typing import Literal
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer, PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel
from peft import LoraConfig, get_peft_model, PeftModel

import torch


def build_clip(
        clip_model_name: Literal['B32', 'B16', 'L', 'H', 'G', 'meta-large', 'meta-huge'],
        mixed_precision: Literal['fp16', 'fp32'] = 'fp32',
        cache_dir: str = ".cache",
):
    clip_model_dict = {'B32': 'openai/clip-vit-base-patch32',
                       'B16': 'openai/clip-vit-base-patch16',
                       'L': 'openai/clip-vit-large-patch14',
                       'H': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                       'G': 'Geonmo/CLIP-Giga-config-fixed',
                       'meta-large': 'facebook/metaclip-l14-fullcc2.5b',
                       'meta-huge': 'facebook/metaclip-h14-fullcc2.5b',
                       }
    
    print(f"Building CLIP model: {clip_model_dict[clip_model_name]}")

    clip_preprocess = CLIPImageProcessor(crop_size={'height': 224, 'width': 224},
                                         do_center_crop=True,
                                         do_convert_rgb=True,
                                         do_normalize=True,
                                         do_rescale=True,
                                         do_resize=True,
                                         image_mean=[0.48145466, 0.4578275, 0.40821073],
                                         image_std=[0.26862954, 0.26130258, 0.27577711],
                                         resample=3,
                                         size={'shortest_edge': 224},
                                         )

    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_dict[clip_model_name], dtype=torch.float16 if mixed_precision == 'fp16' else torch.float32, cache_dir=cache_dir)

    clip_text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_dict[clip_model_name], dtype=torch.float16 if mixed_precision == 'fp16' else torch.float32, cache_dir=cache_dir)

    tokenizer = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder='tokenizer_2', cache_dir=cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens':["[$]"]}) # NOTE: 49408

    return clip_vision_model, clip_preprocess, clip_text_model, tokenizer

class TwoEncoderVLMConfig(PretrainedConfig):
    model_type = "two_encoder_vlm"

    def __init__(
            self, 
            logit_scale: float = 100, 
            trainable_temp: bool = False, 
            encoder_to_freeze: Literal['vision', 'text', 'none'] = 'none',
            alpha_slerp: float = 0.8,
            **kwargs
    ):
        
        super().__init__(**kwargs)
        self.logit_scale = logit_scale
        self.trainable_temp = trainable_temp
        self.encoder_to_freeze = encoder_to_freeze
        self.alpha_slerp = alpha_slerp

class LogitScaleModule(torch.nn.Module):
    # We wrap the parameter in a module to be able to use PEFT
    def __init__(self, init_value):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.log(init_value))

    def forward(self, x=None):
        return self.logit_scale.exp()

class TwoEncoderVLM(PreTrainedModel):
    config_class = TwoEncoderVLMConfig
    base_model_prefix = "two_encoder_vlm"

    def __init__(self, config: TwoEncoderVLMConfig, vision_model, text_model, image_processor=None, tokenizer=None):
        super().__init__(config)
        self.vision = vision_model
        self.text = text_model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.config = config
        self.logit_scale = torch.tensor([config.logit_scale])
        if self.config.trainable_temp:
            self.logit_scale_module = LogitScaleModule(self.logit_scale)

        unfrozen_modules = []
        if self.config.encoder_to_freeze == 'vision':
            for param in self.vision.parameters():
                param.requires_grad = False
        else:
            unfrozen_modules.append('vision')

        if self.config.encoder_to_freeze == 'text':
            for param in self.text.parameters():
                param.requires_grad = False
        else:
            unfrozen_modules.append('text')

        self.unfrozen_modules_prefix = '|'.join(unfrozen_modules)

    def forward(self, pixel_values, input_ids, attention_mask=None, image_name=None, **kwargs):
        vision_out = self.vision(pixel_values=pixel_values)
        text_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        if self.config.trainable_temp:
            logit_scale = self.logit_scale_module(None)
        else:
            logit_scale = self.logit_scale
        return {
            "vision_embeds": vision_out.image_embeds,
            "text_embeds": text_out.text_embeds,
            "vision_outputs": vision_out,
            "text_outputs": text_out,
            "logit_scale": logit_scale,
            "alpha_slerp": self.config.alpha_slerp
        }
    
    def create_peft_model(
            self, 
            lora_config: LoraConfig, 
            adapter_name: str ="adapter",
            adapter_type: Literal ['all', 'no_proj'] = 'all',
        ) -> PeftModel:
        lora_config = copy(lora_config)
        if lora_config.modules_to_save is None:
            lora_config.modules_to_save = []
        if self.config.trainable_temp:
            lora_config.modules_to_save = lora_config.modules_to_save + ["logit_scale_module"] 
        if adapter_type == 'no_proj':
            lora_config.modules_to_save = lora_config.modules_to_save + ["text_projection", "visual_projection"]
        
        if lora_config.task_type is None:
            lora_config.task_type = "FEATURE_EXTRACTION"

        lora_config.target_modules = rf"({self.unfrozen_modules_prefix}).*(q_proj|k_proj|v_proj|out_proj|fc1|fc2|text_projection|visual_projection|position_embedding|token_embedding|patch_embedding)"    

        model_peft = get_peft_model(self, lora_config, adapter_name=adapter_name)
        return model_peft
    
    def apply_peft_from_pretrained(
            self,
            adapter_id: str | PathLike,
            adapter_name: str ="adapter"
        ) -> PeftModel:
        model_peft = PeftModel.from_pretrained(self, adapter_id, adapter_name=adapter_name)
        return model_peft
    
    
class AutoTwoEncoderVLMConfig(TwoEncoderVLMConfig):
    model_type = "auto_two_encoder_vlm"

    def __init__(
            self, 
            model_name: Literal['CLIP_B32', 'CLIP_B16', 'CLIP_L', 'CLIP_H', 'CLIP_G', 'CLIP_meta-large', 'CLIP_meta-huge'] = 'CLIP_B32',
            mixed_precision: Literal['fp16', 'fp32'] = 'fp32',
            cache_dir: str = ".cache",
            **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.mixed_precision = mixed_precision
        self.cache_dir = cache_dir
    
class AutoTwoEncoderVLM(TwoEncoderVLM):
    config_class = AutoTwoEncoderVLMConfig
    base_model_prefix = "auto_two_encoder_vlm"

    def __init__(self, config: AutoTwoEncoderVLMConfig):
        model_name = config.model_name

        if model_name.startswith("CLIP_"):
            clip_model_name = model_name.replace("CLIP_", "")
            vision_model, image_processor, text_model, tokenizer = build_clip(
                clip_model_name=clip_model_name,
                mixed_precision=config.mixed_precision,
                cache_dir=config.cache_dir
            )
        super().__init__(config, vision_model, text_model, image_processor=image_processor, tokenizer=tokenizer)
        self.config = config

AutoConfig.register("auto_two_encoder_vlm", AutoTwoEncoderVLMConfig)
AutoModel.register(AutoTwoEncoderVLMConfig, AutoTwoEncoderVLM)