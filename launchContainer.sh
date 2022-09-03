#!/bin/bash
set -eux
PWD_REAL="$(readlink --canonicalize $PWD)"
echo "PWD_REAL=$PWD_REAL"
docker run --gpus all -it --rm --shm-size=8g \
  -e PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.6,max_split_size_mb:128 \
  -e TZ=JST-9 \
  -v $PWD_REAL:/stable-diffusion/host \
  --hostname sd-docker \
  tateisu/stable-diffusion:0.1
