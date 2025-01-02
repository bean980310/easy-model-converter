import importlib

import torch

from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt, download_controlnet_from_original_ckpt

from convert_lora_safetensor_to_diffusers_custom import convert, convert_xl
from convert_vae_pt_to_diffusers_custom import vae_pt_to_vae_diffuser_xl_base, vae_pt_to_vae_diffuser_xl_refiner
from convert_vae_pt_to_diffusers import vae_pt_to_vae_diffuser

def convert_sd_model(checkpoint_path: str, original_config_file: str=None, config_files: str=None, num_in_channels: int=None, schedler_type: str='pndm', pipeline_type: str=None, image_size: int=None, prediction_type: str=None, extract_ema: bool=True, upcast_attention: bool=False, from_safetensors: bool=True, to_safetensors: bool=True, dump_path: str="./models/diffusers", device: str= 'cuda' if torch.cuda.is_available() else 'cpu', stable_unclip: str=None, stable_unclip_prior: str=None, clip_stats_path: str=None, controlnet: bool=False, half: bool=False, vae_path: str=None, pipeline_class_name: str=None):
    pipeline_class=None

    if pipeline_class_name is not None:
        library = importlib.import_module("diffusers")
        class_obj = getattr(library, pipeline_class_name)
        pipeline_class = class_obj
        
    pipe=download_from_original_stable_diffusion_ckpt(
        checkpoint_path,
        original_config_file=original_config_file,
        config_files=config_files,
        image_size=image_size,
        prediction_type=prediction_type,
        model_type=pipeline_type,
        extract_ema=extract_ema,
        scheduler_type=schedler_type,
        num_in_channels=num_in_channels,
        upcast_attention=upcast_attention,
        from_safetensors=from_safetensors,
        device=device,
        stable_unclip=stable_unclip,
        stable_unclip_prior=stable_unclip_prior,
        clip_stats_path=clip_stats_path,
        controlnet=controlnet,
        vae_path=vae_path,
        pipeline_class=pipeline_class,
    )
    
    if half:
        pipe.to(torch_dtype=torch.float16)
    
    if controlnet:
        pipe.controlnet.save_pretrained(dump_path, safe_serialization=to_safetensors)
    else:
        pipe.save_pretrained(dump_path, safe_serialization=to_safetensors)

def convert_controlnet(checkpoint_path: str, original_config_file: str, num_in_channels: int=None, image_size: int=512, extract_ema: bool=True, upcast_attention: bool=False, from_safetensors: bool=True, to_safetensors: bool=True, dump_path: str="./models/diffusers/controlnet", device: str= 'cuda' if torch.cuda.is_available() else 'cpu', use_linear_projection: bool=False, cross_attention_dim: bool=False):
    controlnet=download_controlnet_from_original_ckpt(
        checkpoint_path=checkpoint_path,
        original_config_file=original_config_file,
        image_size=image_size,
        extract_ema=extract_ema,
        num_in_channels=num_in_channels,
        upcast_attention=upcast_attention,
        from_safetensors=from_safetensors,
        device=device,
        use_linear_projection=use_linear_projection,
        cross_attention_dim=cross_attention_dim,
    )
    controlnet.save_pretrained(dump_path, safe_serialization=to_safetensors)

def convert_lora(base_model_path: str, checkpoint_path: str, dump_path: str="./models/diffusers/loras", lora_prefix_unet: str="lora_unet", lora_prefix_text_encoder: str="lora_te", alpha: int=0.75, to_safetensors: bool=True, device: str= 'cuda' if torch.cuda.is_available() else 'cpu'):
    pipe=convert(base_model_path, checkpoint_path, lora_prefix_unet, lora_prefix_text_encoder, alpha)
    pipe.to(device)
    pipe.save_pretrained(dump_path, safe_serialization=to_safetensors)
    
def convert_lora_xl(base_model_path: str, checkpoint_path: str, dump_path: str="./models/diffusers/loras", lora_prefix_unet: str="lora_unet", lora_prefix_text_encoder: str="lora_te", alpha: int=0.75, to_safetensors: bool=True, device: str= 'cuda' if torch.cuda.is_available() else 'cpu'):
    pipe=convert_xl(base_model_path, checkpoint_path, lora_prefix_unet, lora_prefix_text_encoder, alpha)
    pipe.to(device)
    pipe.save_pretrained(dump_path, safe_serialization=to_safetensors)
    
def convert_vae(vae_path:str, dump_path:str="./models/diffusers/vae"):
    vae_pt_to_vae_diffuser(vae_path, dump_path)
    
def convert_vae_xl_base(vae_path:str, dump_path:str="./models/diffusers/vae"):
    vae_pt_to_vae_diffuser_xl_base(vae_path, dump_path)
    
def convert_vae_xl_refiner(vae_path:str, dump_path:str="./models/diffusers/vae"):
    vae_pt_to_vae_diffuser_xl_refiner(vae_path, dump_path)