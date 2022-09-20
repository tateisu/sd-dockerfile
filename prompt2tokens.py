#!/usr/bin/env python
"""
Get the number of tokens using the same tokenizer that Stable Diffusion uses.

author: opparco
"""

import argparse, sys, re
from transformers import CLIPTokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prompt",
    type=str,
    nargs="?",
    default="a painting of a virus monster playing guitar",
    help="the prompt to render"
)
parser.add_argument(
    "--from-file",
    type=str,
    help="if specified, load prompts from this file",
)
opt = parser.parse_args()

if not opt.from_file:
    prompt = opt.prompt
    assert prompt is not None
    prompts = [prompt]
else:
    print(f"reading prompts from {opt.from_file}")
    with open(opt.from_file, "r") as f:
        prompts = f.read().splitlines()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
max_tokens = tokenizer.model_max_length

for prompt in prompts:
    tokens = tokenizer.tokenize(prompt)
    tokensA = tokens[:max_tokens]
    tokensB = tokens[max_tokens:]
    tokenString = "⁞".join(map(lambda it:re.sub("</w>","",it),tokensA))
    if tokensB:
        tokenStringOverflow = "⁞".join(map(lambda it:re.sub("</w>","",it),tokensB))
        tokenString =f"{tokenString}⁞<max_tokens>⁞{tokenStringOverflow}"

    print(f"prompt({len(tokens)}/{tokenizer.model_max_length}): {tokenString}")
    if len(tokens) > tokenizer.model_max_length :
        print("Too long tokens.")
        sys.exit(1)
