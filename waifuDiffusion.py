import os, sys, torch
from torch import autocast
from diffusers import StableDiffusionPipeline

device = "cuda"
model_id = "hakurei/waifu-diffusion"
model_dir = "host/model-waifu-diffusion"
prompt = "touhou hakurei_reimu 1girl solo portrait"

if os.path.exists(f"{model_dir}/model_index.json"):
    # ディレクトリから読む
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir, 
        torch_dtype=torch.float16, 
        revision='fp16',
    )
else:
    # リポジトリから読む
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        revision='fp16',
    )
    # ローカルに保存
    print(f"save {model_id} to {model_dir}…")
    os.makedirs(model_dir, exist_ok=True)
    pipe.save_pretrained(save_directory=model_dir)

# pipeを使ってtxt2img
pipe = pipe.to(device)
with autocast(device):
    image = pipe(
        prompt, 
        guidance_scale=7.5,
    )["sample"][0]  
    image.save("host/reimu_hakurei2.png")
