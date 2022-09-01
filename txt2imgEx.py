import argparse, os, re
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
from transformers import logging
import pandas as pd
import datetime
import json
from collections import OrderedDict
import codecs
from pprint import pprint

logging.set_verbosity_error()


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

# グリッド内の枚数。
parser.add_argument(
    "--n_samples",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)

parser.add_argument(
    "--fixed_code",
    action="store_true",
    help="if enabled, uses the same starting code across samples ",
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
parser.add_argument(
    "--n_rows",
    type=int,
    default=0,
    help="rows in the grid (default: n_samples)",
)
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="specify GPU (cuda/cuda:0/cuda:1/...)",
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

opt = parser.parse_args()

tic = time.time()
outpath = opt.outdir
os.makedirs(outpath, exist_ok=True)
grid_count = len(os.listdir(outpath)) - 1

fixed_seed = opt.seed != None

if opt.seed == None:
    opt.seed = randint(1,2147483647)
seed_everything(opt.seed)

# Logging
logger(vars(opt), log_csv = "logs/txt2img_logs.csv")

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
model.cdevice = opt.device
model.turbo = opt.turbo

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = opt.device

modelFS = instantiate_from_config(config.modelFirstStage)
_, _ = modelFS.load_state_dict(sd, strict=False)
modelFS.eval()
del sd

is_half = False
if opt.device != "cpu" and opt.precision == "autocast":
    is_half = True
    model.half()
    modelCS.half()

start_code = None
if opt.fixed_code:
    start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=opt.device)


batch_size = opt.n_samples
n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
if not opt.from_file:
    prompt = opt.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    pprint(data)

else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        data = f.read().splitlines()
        data = batch_size * list(data)
        data = list(chunk(sorted(data), batch_size))


if opt.precision == "autocast" and opt.device != "cpu":
    precision_scope = autocast
else:
    precision_scope = nullcontext

print( "{0:.2f} seconds for loading.".format(time.time() - tic) )

with torch.no_grad():

    for prompts in tqdm(data, desc="data"):

        tic = time.time()

        with precision_scope("cuda"):
            modelCS.to(opt.device)
            uc = None
            if opt.scale != 1.0:
                uc = modelCS.get_learned_conditioning(batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)

            subprompts, weights = split_weighted_subprompts(prompts[0])
            pprint(weights)
            pprint(subprompts)
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

            print( "{0:.2f} seconds for modelCS.".format(time.time() - tic) )
            for repeat_remain in reversed(range(opt.repeat)):

                print("================================")
                print(f"repeat {opt.repeat-repeat_remain}/{opt.repeat}")

                if opt.device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    print(f"modelCS.to(cpu) cuda_allocated={mem}")
                    modelCS.to("cpu")
                    while mem > 1000 and torch.cuda.memory_allocated() / 1e6 >= mem:
                        print( "waiting cuda memory…alocated={0:.2f}".format(torch.cuda.memory_allocated() / 1e6) )
                        time.sleep(1)

                tic = time.time()

                samples_ddim = model.sample(
                    S=opt.ddim_steps,
                    conditioning=c,
                    batch_size=opt.n_samples,
                    seed=opt.seed,
                    shape=shape,
                    verbose=False,
                    unconditional_guidance_scale=opt.scale,
                    unconditional_conditioning=uc,
                    eta=opt.ddim_eta,
                    x_T=start_code,
                )

                modelFS.to(opt.device)

                print(samples_ddim.shape)
                print("saving images")

                seeds = ""

                for i in range(batch_size):

                    x_samples_ddim = modelFS.decode_first_stage(samples_ddim[i].unsqueeze(0))
                    x_sample = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_sample = 255.0 * rearrange(x_sample[0].cpu().numpy(), "c h w -> h w c")
                    time_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                    basename = os.path.join(outpath, f"{time_str}_{opt.seed}")

                    image = Image.fromarray(x_sample.astype(np.uint8))
                    image.save(f"{basename}.{opt.format}")

                    info = OrderedDict()
                    info['text'] = opt.prompt
                    info['folder'] = outpath
                    info['resX'] = image.width
                    info['resY'] = image.height
                    info['half'] = is_half
                    info['seed'] = opt.seed
                    info['steps'] = opt.ddim_steps
                    info['vscale'] = opt.scale
                    info['C'] = opt.C
                    info['ckpt'] = os.path.basename(os.readlink( ckpt ))
                    info_json = json.dumps(info, ensure_ascii=False)
                    f = codecs.open(f"{basename}_info.txt", 'w', 'utf-8')
                    f.write(info_json)
                    f.close()

                    seeds += str(opt.seed) + ","
                    opt.seed += 1

                print("image saved. seeds=" + seeds[:-1])
                print( "{0:.2f} seconds for samples_ddim.".format(time.time() - tic) )

                del samples_ddim

                if opt.device != "cpu":
                    mem = torch.cuda.memory_allocated() / 1e6
                    print(f"modelFS.to(cpu) cuda_allocated={mem}")
                    modelFS.to("cpu")
                    while mem > 1000 and torch.cuda.memory_allocated() / 1e6 >= mem:
                        print( f"waiting cuda memory…alocated={0:.2f}".format(torch.cuda.memory_allocated() / 1e6) )
                        time.sleep(1)

                print("cuda_allocated=", torch.cuda.memory_allocated() / 1e6)
                
                if fixed_seed and repeat_remain>0:
                    print( "repeat is ignored for fixed_seed")
                    break

                opt.seed = randint(1,2147483647)
