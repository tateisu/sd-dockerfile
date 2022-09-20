#!/usr/bin/env python

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
from distutils.util import strtobool

import pandas as pd
import datetime
import json
from collections import OrderedDict
import codecs
from pprint import pprint
import torch.nn as nn
import k_diffusion as K

logging.set_verbosity_error()

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
    "--emojiMap",
    type=str,
    help="input file of emoji list.",
)

device = "cpu"

opt = parser.parse_args()

##############################
# load model

tic = time.time()

seed_everything(randint(1,2147483647))

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
model.unet_bs = 1
model.cdevice = device
model.turbo = False

modelCS = instantiate_from_config(config.modelCondStage)
_, _ = modelCS.load_state_dict(sd, strict=False)
modelCS.eval()
modelCS.cond_stage_model.device = device

tokenizer = modelCS.cond_stage_model.tokenizer
max_tokens = tokenizer.model_max_length -2

print(f"{time.time()-tic:.2f}s for loading model and tokenizer.")

def escape(a):
    return re.sub(r'(\\|[^\x21-\x7e])', lambda m: f"\\u{ord(m.group(1)):04}", a)

CEND = '\33[0m'
CRED = '\33[31m'
def checkPrompt(caption,prompt):
    tokens = tokenizer.tokenize(prompt)
    tokensA = list(map(escape,tokens[:max_tokens]))
    tokensB = list(map(escape,tokens[max_tokens:]))
    tokenString = "⁞".join(map(lambda it:re.sub("</w>","",it),tokensA))
    if tokensB:
        tokenStringOverflow = "⁞".join(map(lambda it:re.sub("</w>","",it),tokensB))
        tokenString =f"{tokenString}⁞{CRED}<max_tokens>{CEND}⁞{tokenStringOverflow}"
    print(f"emoji={caption}, tokens={len(tokens)}, prompt={tokenString}")

with open(opt.emojiMap,encoding='utf-8')as f:
    while True:
        line = f.readline()
        if line == '':
            break
        line = re.sub(r'[\x0d\x0a]+', '', line)
        line = re.sub(r'//.*', '', line)
        m = re.search(r'^un:(.+)', line)
        if m:
            text = m.group(1)
            checkPrompt(text,text)
