import argparse, os, re, sys
import torch
import numpy as np
from random import randint
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ldm.util import instantiate_from_config
from optimUtils import split_weighted_subprompts, logger
from transformers import logging, CLIPTokenizer

import pandas as pd
import datetime
import json
from collections import OrderedDict
import codecs
from pprint import pprint
import torch.nn as nn
import k_diffusion as K

logging.set_verbosity_error()

def formatBytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)
    GB = float(KB ** 3)
    TB = float(KB ** 4)
    if   B>=TB: return f"{B/TB:.1f}TB"
    elif B>=GB: return f"{B/GB:.1f}GB"
    elif B>=MB: return f"{B/MB:.1f}MB"
    elif B>=KB: return f"{B/KB:.1f}KB"
    else:       return f"{B} byte(s)"

def free_vram(caption, block):
    before = torch.cuda.memory_allocated()
    print(f"CUDA allocated={formatBytes(before)}. freeing {caption}…")
    block()
    while before > 1_000_000_000 and torch.cuda.memory_allocated() >= before-1_000_000:
        print( "CUDA allocated={formatBytes(torch.cuda.memory_allocated())}. waiting free…")
        time.sleep(3)

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def load_model_from_config(ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    return sd

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale

def create_random_tensors(shape, seed, device):
    xs = []
    torch.manual_seed(seed)
    xs.append(torch.randn(shape, device=device))
    x = torch.stack(xs, 0)
    return x

config = "host/v1-inference.yaml"
ckpt = "models/ldm/stable-diffusion-v1/model.ckpt"

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str, 
    nargs="?", 
    default="a painting of a virus monster playing guitar", 
    help="the prompt to render"
)
parser.add_argument(
    "--outdir", 
    type=str, 
    nargs="?", 
    help="dir to write results to", 
    default="outputs"
)
parser.add_argument(
    "--skip_grid",
    action="store_true",
    help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
)

parser.add_argument(
    "--skip_save",
    action="store_true",
    help="do not save individual samples. For speed measurements.",
)

# GRisk GUI のStepsに相当する？
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=50,
    help="number of ddim sampling steps",
)

parser.add_argument(
    "--fixed_code",
    action="store_true",
    help="if enabled, uses the same starting code across samples",
)

parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)

parser.add_argument(
    "--H",
    type=int,
    default=512,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=512,
    help="image width, in pixel space",
)

parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)

parser.add_argument(
    "--f",
    type=int,
    default=8,
    help="downsampling factor",
)

# プロンプト指定への忠実度
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="the seed (for reproducible sampling)",
)

parser.add_argument(
    "--unet_bs",
    type=int,
    default=1,
    help="Slightly reduces inference time at the expense of high VRAM (value > 1 not recommended )",
)

parser.add_argument(
    "--turbo",
    action="store_true",
    help="Reduces inference time on the expense of 1GB VRAM",
)

parser.add_argument(
    "--precision", 
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)

parser.add_argument(
    "--format",
    type=str,
    help="output image format",
    choices=["jpg", "png"],
    default="png",
)
parser.add_argument(
    "--repeat",
    type=int,
    help="if >1, repeat image creation with random seed. if seed is fixed, this option has no effect.",
    default=1,
)
parser.add_argument(
    "--cooldown",
    type=int,
    help="if >0, sleep specified seconds before each image creation.",
    default=0,
)
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler. one of k_euler_a, k_dpm_2_a, k_lms(default)",
    choices=["k_euler_a", "k_dpm_2_a", "k_lms"],
    default="k_lms"
)
parser.add_argument(
    "--allow-long-token",
    action="store_true",
    help="it true, don't check token length.",
)

opt = parser.parse_args()

device = "cuda"

tic = time.time()
outpath = opt.outdir
os.makedirs(outpath, exist_ok=True)

is_fixed_seed = opt.seed != None

if opt.seed == None:
    opt.seed = randint(1,2147483647)
seed_everything(opt.seed)

sd = load_model_from_config(f"{ckpt}")
li, lo = [], []
for key, value in sd.items():
    sp = key.split(".")
    if (sp[0]) == "model":
        if "input_blocks" in sp:
            li.append(key)
        elif "middle_block" in sp:
            li.append(key)
        elif "time_embed" in sp:
            li.append(key)
        else:
            lo.append(key)
for key in li:
    sd["model1." + key[6:]] = sd.pop(key)
for key in lo:
    sd["model2." + key[6:]] = sd.pop(key)

config = OmegaConf.load(f"{config}")

model = instantiate_from_config(config.modelUNet)
_, _ = model.load_state_dict(sd, strict=False)
model.eval()
model.unet_bs = opt.unet_bs
model.cdevice = device
model.turbo = opt.turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd

ksamplers = {
    'k_lms': K.sampling.sample_lms, 
    'k_euler_a': K.sampling.sample_euler_ancestral, 
    'k_dpm_2_a': K.sampling.sample_dpm_2_ancestral,
}

model_wrap = K.external.CompVisDenoiser(model)
sampler = ksamplers[opt.sampler]

is_half = False
if opt.precision == "autocast":
    is_half = True
    model.half()
    modelCS.half()

if not opt.from_file:
    prompt = opt.prompt
    assert prompt is not None
    data = [1 * [prompt]]
else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = 1 * list(data)
        data = list(chunk(sorted(data), 1))

if opt.precision == "autocast" :
    precision_scope = autocast
else:
    precision_scope = nullcontext

print(f"{time.time()-tic:.2f}s for loading model.")

tic = time.time()
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
print(f"{time.time()-tic:.2f}s for loading tokenizer for validation.")

max_tokens = tokenizer.model_max_length
# 上限は77だが、ギリギリに置いた単語の効果は出ないっぽい。
# https://note.com/hugiri/n/n970f9deb55b2

with torch.no_grad():
    for prompts in data:

        tic = time.time()

        with precision_scope("cuda"):
            modelCS.to(device)
            uc = None
            if opt.scale != 1.0:
                uc = modelCS.get_learned_conditioning(1 * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            subprompts, weights = split_weighted_subprompts(prompts[0])
            
            for i,prompt in enumerate(subprompts):
                tokens = tokenizer.tokenize(prompt)
                print(f"prompt[{i}]: {len(tokens)}/{max_tokens} tokens. weight={weights[i]}, prompt={prompt}")
                if not opt.allow_long_token and len(tokens) > max_tokens :
                    print(f"prompt[{i}]: Too long tokens.")
                    sys.exit(1)

            if len(subprompts) > 1:
                c = torch.zeros_like(uc)
                totalWeight = sum(weights)
                # normalize each "sub prompt" and add it
                for i in range(len(subprompts)):
                    weight = weights[i]
                    # if not skip_normalize:
                    weight = weight / totalWeight
                    c = torch.add(c, modelCS.get_learned_conditioning(subprompts[i]), alpha=weight)
            else:
                c = modelCS.get_learned_conditioning(prompts)

            shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

            free_vram("modelCS", lambda: modelCS.to("cpu"))

            print(f"{time.time() - tic:.2f}s for prompt conversion.")

            for repeat_remain in reversed(range(opt.repeat)):

                print("================================")
                print(f"repeat {opt.repeat-repeat_remain}/{opt.repeat}")

                if opt.cooldown > 0:
                    time.sleep(opt.cooldown)

                tic = time.time()

                model_wrap.to(device)

                sigmas = model_wrap.get_sigmas(opt.ddim_steps)
                x = create_random_tensors(
                    shape, 
                    opt.seed, 
                    device=device,
                ) * sigmas[0]

                extra_args = {
                    'cond': c, 
                    'uncond': uc, 
                    'cond_scale': opt.scale
                }

                samples = sampler(
                    CFGDenoiser(model_wrap),
                    x, 
                    sigmas, 
                    extra_args=extra_args, 
                    disable=False
                )

                modelFS.to(device)

                x_samples_ddim = modelFS.decode_first_stage(samples[0].unsqueeze(0))
                x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                image = Image.fromarray(x_sample.astype(np.uint8))

                print(f"{time.time()-tic:.2f}s to compute samples. shape={samples.shape}")
                del samples

                time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                basename = os.path.join(outpath, f"{time_str}_{opt.seed}")
                imageFile=f"{basename}.{opt.format}"
                print(f"save to {imageFile}")
                image.save(imageFile)

                info = OrderedDict()
                info['text'] = opt.prompt
                info['folder'] = outpath
                info['resX'] = image.width
                info['resY'] = image.height
                info['half'] = is_half
                info['seed'] = opt.seed
                info['steps'] = opt.ddim_steps
                info['scale'] = opt.scale
                info['C'] = opt.C
                info['ckpt'] = os.path.basename(os.readlink( ckpt ))
                info['sampler'] = opt.sampler
                info_json = json.dumps(info, ensure_ascii=False)
                f = codecs.open(f"{basename}_info.txt", 'w', 'utf-8')
                f.write(info_json)
                f.close()

                free_vram("modelFS", lambda: modelFS.to("cpu"))
                free_vram("model_wrap", lambda: model_wrap.to("cpu"))
                # 不要ぽい free_vram("modelCS", lambda: modelCS.to("cpu"))

                # リーク検出のため、1バイト単位で表示する
                print(f"CUDA allocated={torch.cuda.memory_allocated():,} bytes")

                if is_fixed_seed and repeat_remain>0:
                    print( "repeat is ignored for fixed seed.")
                    break

                opt.seed = randint(1,2147483647)
