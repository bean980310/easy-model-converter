import argparse
import io

import requests
import torch
import yaml

from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
    assign_to_checkpoint,
    conv_attn_to_linear,
    create_vae_diffusers_config,
    renew_vae_attention_paths,
    renew_vae_resnet_paths,
)

from convert_vae_pt_to_diffusers import custom_convert_ldm_vae_checkpoint

def vae_pt_to_vae_diffuser_xl_base(
    checkpoint_path: str,
    output_path: str,
):
    # Only support XL Base
    r = requests.get(
        "https://raw.githubusercontent.com/Stability-AI/generative-models/refs/heads/main/configs/inference/sd_xl_base.yaml"
    )
    io_obj = io.BytesIO(r.content)

    original_config = yaml.safe_load(io_obj)
    image_size = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path.endswith("safetensors"):
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    vae.save_pretrained(output_path)
    
def vae_pt_to_vae_diffuser_xl_refiner(
    checkpoint_path: str,
    output_path: str,
):
    # Only support XL Refiner
    r = requests.get(
        "https://raw.githubusercontent.com/Stability-AI/generative-models/refs/heads/main/configs/inference/sd_xl_refiner.yaml"
    )
    io_obj = io.BytesIO(r.content)

    original_config = yaml.safe_load(io_obj)
    image_size = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if checkpoint_path.endswith("safetensors"):
        from safetensors import safe_open

        checkpoint = {}
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                checkpoint[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)["state_dict"]

    # Convert the VAE model.
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = custom_convert_ldm_vae_checkpoint(checkpoint, vae_config)

    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(converted_vae_checkpoint)
    vae.save_pretrained(output_path)