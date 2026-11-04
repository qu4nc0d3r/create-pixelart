"""SDXL + pixel-art LoRA generation on the Kaggle GPU."""
import random
from contextlib import contextmanager

import torch

from pixelart.postprocess import prepare_input

BASE = "stabilityai/stable-diffusion-xl-base-1.0"
LORA = "nerijs/pixel-art-xl"
VAE = "madebyollin/sdxl-vae-fp16-fix"
PREFIX = "pixel art, "
NEGATIVE = "blurry, 3d render, photo, realistic, gradient, noise"
STEPS = 30
GUIDANCE = 7.0

_txt2img = None
_img2img = None


def resolve_seed(seed):
    if seed is None or seed < 0:
        return random.randint(0, 2**31 - 1)
    return int(seed)


@contextmanager
def seamless(*modules):
    convs = [m for mod in modules for m in mod.modules() if isinstance(m, torch.nn.Conv2d)]
    old = [c.padding_mode for c in convs]
    for c in convs:
        c.padding_mode = "circular"
    try:
        yield
    finally:
        for c, mode in zip(convs, old):
            c.padding_mode = mode


def load():
    global _txt2img, _img2img
    from diffusers import AutoencoderKL, AutoPipelineForImage2Image, StableDiffusionXLPipeline

    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=torch.float16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, vae=vae, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.load_lora_weights(LORA)
    pipe.vae.enable_slicing()  # decode one image at a time; n=4 at 1024² would OOM a T4
    _txt2img = pipe
    _img2img = AutoPipelineForImage2Image.from_pipe(pipe)  # shares weights, no extra VRAM


def _run(pipe, prompt, n, seed, **kwargs):
    seed = resolve_seed(seed)
    result = pipe(
        prompt=PREFIX + prompt,
        negative_prompt=NEGATIVE,
        num_inference_steps=STEPS,
        guidance_scale=GUIDANCE,
        num_images_per_prompt=n,
        generator=[torch.Generator("cuda").manual_seed(seed + i) for i in range(n)],
        **kwargs,
    )
    return result.images, seed


def txt2img(prompt, n, seed):
    return _run(_txt2img, prompt, n, seed, width=1024, height=1024)


def img2img(image, prompt, strength, n, seed):
    return _run(_img2img, prompt, n, seed, image=prepare_input(image), strength=strength)


def tile(prompt, n, seed):
    with seamless(_txt2img.unet, _txt2img.vae):
        return txt2img(prompt, n, seed)
