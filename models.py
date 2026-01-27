#Adatapted from https://github.com/jaeseokbyun/MACIR/blob/main/models.py

from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPImageProcessor, CLIPTokenizer,BlipForImageTextRetrieval
import torch


def build_clip(args):
    clip_model_dict = {'B32': 'openai/clip-vit-base-patch32',
                       'B16': 'openai/clip-vit-base-patch16',
                       'L': 'openai/clip-vit-large-patch14',
                       'H': 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K',
                       'G': 'Geonmo/CLIP-Giga-config-fixed',
                       'meta-large': 'facebook/metaclip-l14-fullcc2.5b',
                       'meta-huge': 'facebook/metaclip-h14-fullcc2.5b',
                       }
    
    print(args)
    print(f"Building CLIP model: {clip_model_dict[args.clip_model_name]}")

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

    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_model_dict[args.clip_model_name], dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.float32, cache_dir=args.cache_dir)

    clip_text_model = CLIPTextModelWithProjection.from_pretrained(clip_model_dict[args.clip_model_name], dtype=torch.float16 if args.mixed_precision == 'fp16' else torch.float32, cache_dir=args.cache_dir)

    tokenizer = CLIPTokenizer.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', subfolder='tokenizer_2', cache_dir=args.cache_dir)
    tokenizer.add_special_tokens({'additional_special_tokens':["[$]"]}) # NOTE: 49408

    return clip_vision_model, clip_preprocess, clip_text_model, tokenizer

class TwoEncoderVLM(torch.nn.Module):
    def __init__(self, vision_model, text_model, logit_scale, proj_dim=None, trainable_temp=False, image_processor=None, tokenizer=None):
        super().__init__()
        self.vision = vision_model
        self.text = text_model
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.proj_dim = proj_dim
        self.logit_scale = logit_scale
        self.trainable_temp = trainable_temp
        if self.trainable_temp:
            self.s = torch.nn.Parameter(torch.Tensor([logit_scale]).log())

    def forward(self, pixel_values, input_ids, attention_mask=None, image_name=None, **kwargs):
        vision_out = self.vision(pixel_values=pixel_values)
        text_out = self.text(input_ids=input_ids, attention_mask=attention_mask)
        if self.trainable_temp:
            logit_scale = self.s.exp()
        else:
            logit_scale = self.logit_scale
        return {
            "vision_embeds": vision_out.image_embeds,
            "text_embeds": text_out.text_embeds,
            "vision_outputs": vision_out,
            "text_outputs": text_out,
            "logit_scale": logit_scale,
        }