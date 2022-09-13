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
    block()
    while before > 1_000_000_000 and torch.cuda.memory_allocated() >= before-1_000_000:
        print( "CUDA allocated={formatBytes(before)} → {formatBytes(torch.cuda.memory_allocated())}. waiting free {caption}…")
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

# Guidance scale, プロンプト指定への忠実度
parser.add_argument(
    "--scale",
    type=str,
    default="7.5",
    help="comma separated, unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)

# 
parser.add_argument(
    "--steps",
    type=str,
    default="50",
    help="comma separated, number of ddim sampling steps",
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
    "--allow-long-token",
    action="store_true",
    help="it true, don't check token length.",
)
parser.add_argument(
   "--comment",
    type=str,
    help="comment is added to json.",
    default="",
)
parser.add_argument(
    "--tile",
    action="store_true",
    help="make image for tile.",
    default=False
)
parser.add_argument(
    "--negative-prompt",
    type=str,
    help="negative prompt",
    default=""
)
parser.add_argument(
    "--negative-prompt-alpha",
    type=str,
    default="1",
    help="comma separated, alpha(0..1) of negative prompt.",
)

ksamplers = {
    'k_lms': K.sampling.sample_lms, 
    'k_euler_a': K.sampling.sample_euler_ancestral, 
    'k_dpm_2_a': K.sampling.sample_dpm_2_ancestral,
}
parser.add_argument(
    "--sampler",
    type=str,
    help="sampler. one of k_euler_a, k_dpm_2_a, k_lms(default)",
    choices=["k_euler_a", "k_dpm_2_a", "k_lms"],
    default="k_lms"
)

device = "cuda"

opt = parser.parse_args()

def splitCSV(a,typeConverter):
    return list(
        map(
            typeConverter,
            filter(
                lambda x: len(x)>0,
                map(
                    lambda x: x.strip(),
                    a.split(","),
                ),
            ),
        )
    )

# multiple scales, steps
scales = splitCSV(opt.scale, lambda x: float(x))
steps_list = splitCSV(opt.steps, lambda x: int(x))
negative_prompt_alphas = splitCSV(opt.negative_prompt_alpha, lambda x: float(x))

is_fixed_seed = (opt.seed is not None)
if opt.repeat>1 and is_fixed_seed:
    print( "repeat is ignored for fixed seed")
    opt.repeat = 1

##############################
# load model

tic = time.time()

def patch_conv(cls):
    init = cls.__init__
    def __init__(self, *args, **kwargs):
        return init(self, *args, **kwargs, padding_mode='circular')
    cls.__init__ = __init__

if opt.tile:
    patch_conv(torch.nn.Conv2d)

# initialize seed if not specified
if not is_fixed_seed:
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

#######################################
# prompt token check

if opt.from_file:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        promptText = f.read()
else:
    promptText = opt.prompt

promptText = re.sub('[\x00-\x20]+', ' ', promptText)

# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

# modelCS.cond_stage_model は ldm.modules.encoders.modules.FrozenCLIPEmbedder である
# less ldm/modules/encoders/modules.py
tokenizer = modelCS.cond_stage_model.tokenizer

print(f"{time.time()-tic:.2f}s for loading tokenizer for validation.")

# 上限は77だが、ギリギリに置いた単語の効果は出ないっぽい。
# https://note.com/hugiri/n/n970f9deb55b2
max_tokens = tokenizer.model_max_length

def checkPrompt(caption,prompt):
    tokens = tokenizer.tokenize(prompt)
    tokensA = tokens[:max_tokens]
    tokensB = tokens[max_tokens:]
    tokenString = "⁞".join(map(lambda it:re.sub("</w>","",it),tokensA))
    if tokensB:
        tokenStringOverflow = "⁞".join(map(lambda it:re.sub("</w>","",it),tokensB))
        tokenString =f"{tokenString}⁞<max_tokens>⁞{tokenStringOverflow}"

    print(f"{caption}: {len(tokens)}/{max_tokens} tokens. weight={weights[i]}, prompt={tokenString}")
    if not opt.allow_long_token and tokensB:
        print(f"{caption}: Too long tokens.")
        sys.exit(1)

subprompts, weights = split_weighted_subprompts(promptText)
for i,prompt in enumerate(subprompts):
    checkPrompt(f"prompt[{i}]",prompt)

if opt.negative_prompt:
    checkPrompt("negative_prompt",opt.negative_prompt)

######################################

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd

model_wrap = K.external.CompVisDenoiser(model)
sampler = ksamplers[opt.sampler]

is_half = False
if opt.precision == "autocast":
    is_half = True
    model.half()
    modelCS.half()

if opt.precision == "autocast" :
    precision_scope = autocast
else:
    precision_scope = nullcontext

print(f"{time.time()-tic:.2f}s for loading model.")

# create output dir
outpath = opt.outdir
os.makedirs(outpath, exist_ok=True)

tic = time.time()
with torch.no_grad():
    with precision_scope("cuda"):
        for repeat_remain in reversed(range(opt.repeat)):
            for negative_prompt_alpha in negative_prompt_alphas:
                modelCS.to(device)

                tic = time.time()

                # 素のuc。ゼロではない
                uc_base = modelCS.get_learned_conditioning([""])

                # 素のucとネガティブプロンプトを足し合わせる
                uc = torch.zeros_like(modelCS.get_learned_conditioning([""]))
                uc = torch.add(
                    uc,
                    uc_base,
                    alpha= (1.0-negative_prompt_alpha)
                )
                uc = torch.add(
                    uc,
                    modelCS.get_learned_conditioning([opt.negative_prompt]),
                    alpha=negative_prompt_alpha
                )

                # normalize each "sub prompt" and add it
                c = torch.zeros_like(modelCS.get_learned_conditioning([""]))
                totalWeight = sum(weights)
                for i in range(len(subprompts)):
                    weight = weights[i]
                    # if not skip_normalize:
                    weight = weight / totalWeight
                    c = torch.add(
                        c, 
                        modelCS.get_learned_conditioning(subprompts[i]), 
                        alpha=weight
                    )

                shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                print(f"{time.time() - tic:.2f}s for prompt conversion.")

                free_vram("modelCS", lambda: modelCS.to("cpu"))

                for scale in scales:
                    for steps in steps_list:
                        print("================================")
                        print(f"repeat={opt.repeat-repeat_remain}/{opt.repeat}, scale={scale}, steps={steps}, negative_prompt_alpha={negative_prompt_alpha}")

                        if opt.cooldown > 0:
                            time.sleep(opt.cooldown)

                        tic = time.time()

                        model_wrap.to(device)

                        sigmas = model_wrap.get_sigmas(steps)
                        x = create_random_tensors(
                            shape, 
                            opt.seed, 
                            device=device,
                        ) * sigmas[0]

                        extra_args = {
                            'cond': c, 
                            'uncond': uc, 
                            'cond_scale': scale
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
                        if len(scales) > 1:
                            basename = basename + f"_scale{scale}"
                        if len(steps_list) > 1:
                            basename = basename + f"_steps{steps}"
                        if len(negative_prompt_alphas) > 1:
                            basename = basename + f"_npa{negative_prompt_alpha}"

                        imageFile=f"{basename}.{opt.format}"
                        print(f"save to {imageFile}")
                        image.save(imageFile)

                        info = OrderedDict()
                        info['prompt'] = opt.prompt
                        if opt.negative_prompt:
                            info['negative_prompt']=opt.negative_prompt
                            info['negative_prompt_alpha']=negative_prompt_alpha

                        info['ckpt'] = os.path.basename(os.readlink( ckpt ))
                        info['width'] = image.width
                        info['height'] = image.height
                        info['seed'] = opt.seed
                        info['sampler'] = opt.sampler
                        info['steps'] = steps
                        info['scale'] = scale
                        info['precision'] = opt.precision
                        info['C'] = opt.C
                        comment = opt.comment.strip()
                        if comment:
                            info['comment'] = comment
                        info_json = json.dumps(info, ensure_ascii=False)
                        f = codecs.open(f"{basename}_info.txt", 'w', 'utf-8')
                        f.write(info_json)
                        f.close()

                        free_vram("modelFS", lambda: modelFS.to("cpu"))
                        free_vram("model_wrap", lambda: model_wrap.to("cpu"))
                        # 不要ぽい free_vram("modelCS", lambda: modelCS.to("cpu"))

                        # リーク検出のため、1バイト単位で表示する
                        print(f"CUDA allocated={torch.cuda.memory_allocated():,} bytes")

            if not is_fixed_seed:
                opt.seed = randint(1,2147483647)
