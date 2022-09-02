FROM nvcr.io/nvidia/pytorch:22.08-py3

####################################################
# APT packages

RUN apt-get update && \
    apt-get install -y wget git git-lfs libglib2.0-0 libsm6 libxrender1 libxext-dev libvulkan1 && \
	apt-get clean && \
	rm -rf /var/lib/apt/lists/*

####################################################
# Stable Diffusion

# 毎回git cloneするとDockerがキャッシュできないので、取得済みのtgzを展開する
ADD stable-diffusion-src.tgz /stable-diffusion

COPY requirements.txt /stable-diffusion

# omegaconf は別途upgradeをつけてインストールしないと
# WARNING: Package(s) not found: omegaconf で怒られる
RUN cd /stable-diffusion && \
	pip install --upgrade omegaconf>=2.1.1 && \
	pip install -r requirements.txt && \
	pip show omegaconf invisible-watermark

####################################################
# realesrgan

ADD realesrgan.tgz /realesrgan

RUN cd /realesrgan && \
	chmod 755 realesrgan-ncnn-vulkan

####################################################

RUN mkdir -p /stable-diffusion/host

VOLUME /stable-diffusion/host

WORKDIR /stable-diffusion
