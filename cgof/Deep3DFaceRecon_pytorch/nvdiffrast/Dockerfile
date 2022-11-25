# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Note: Should also work with NVIDIA's Docker image builds such as
#
# nvcr.io/nvidia/pytorch:20.09-py3
#
# This file defaults to pytorch/pytorch as it works on slightly older
# driver versions.
# ARG BASE_IMAGE=pytorch/pytorch:1.7.1-cuda11.0-cudnn8-devel
ARG BASE_IMAGE=pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel
FROM $BASE_IMAGE

RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    curl

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# for GLEW
ENV LD_LIBRARY_PATH /usr/lib64:$LD_LIBRARY_PATH

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics

# Default pyopengl to EGL for good headless rendering support
ENV PYOPENGL_PLATFORM egl

COPY 10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

RUN pip install ninja imageio imageio-ffmpeg pynvml loguru einops timm PyMCubes sklearn 

RUN mkdir /tmp/pip
COPY nvdiffrast/nvdiffrast /tmp/pip/nvdiffrast
COPY nvdiffrast/README.md nvdiffrast/setup.py /tmp/pip/
RUN cd /tmp/pip && pip install .

RUN apt-get install -y curl wget vim git
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa

COPY requirements.txt /root/requirement.txt
RUN pip install -r /root/requirement.txt

# install gfpgan
# RUN mkdir /tmp/gfpgan
COPY gfpgan /tmp/gfpgan
RUN pip install basicsr==1.3.5
RUN pip install facexlib==0.2.2
RUN cd /tmp/gfpgan && pip install -r requirements.txt
RUN cd /tmp/gfpgan && python setup.py develop
RUN pip install realesrgan==0.2.5

# change to your project path.
ARG PROJ_PATH=${PROJ_PATH:-/home/kqsun/Tasks/cgof_release/cgof}
RUN echo export PATH=\"\$HOME/bin:\$HOME/.local/bin:\$PATH\" >> ~/.bashrc
RUN echo export PYTHONPATH=\"$PROJ_PATH:\$PYTHONPATH\" >> ~/.bashrc
RUN echo export PYTHONPATH=\"$PROJ_PATH/Deep3DFaceRecon_pytorch:\$PYTHONPATH\" >> ~/.bashrc

RUN conda install -c fvcore -c iopath -c conda-forge fvcore iopath
RUN conda install -c bottler nvidiacub
RUN conda install pytorch3d -c pytorch3d
