# Controllable 3D Face Synthesis with Conditional Generative Occupancy Fields (cGOF)
Official PyTorch implementation for the NeurIPS 2022 paper.

![Avatar](docs/teaser.png)

## Controllable 3D Face Synthesis with Conditional Generative Occupancy Fields
[Keqiang Sun](https://keqiangsun.github.io)*, [Shangzhe Wu](https://elliottwu.com)*, [Zhaoyang Huang](https://drinkingcoder.github.io), Ning Zhang, [Quan Wang](https://scholar.google.com/citations?user=KmxEHm4AAAAJ&hl=zh-TW), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/) \
\* equal contribution

https://keqiangsun.github.io/projects/cgof/

Abstract: Capitalizing on the recent advances in image generation models, existing controllable face image synthesis methods are able to generate high-fidelity images with some levels of controllability, e.g., controlling the shapes, expressions, textures, and poses of the generated face images. However, these methods focus on 2D image generative models, which are prone to producing inconsistent face images under large expression and pose changes. In this paper, we propose a new NeRF-based conditional 3D face synthesis framework, which enables 3D controllability over the generated face images by imposing explicit 3D conditions from 3D face priors. At its core is a conditional Generative Occupancy Field (cGOF) that effectively enforces the shape of the generated face to commit to a given 3D Morphable Model (3DMM) mesh. To achieve accurate control over fine-grained 3D face shapes of the synthesized image, we additionally incorporate a 3D landmark loss as well as a volume warping loss into our synthesis algorithm. Experiments validate the effectiveness of the proposed method, which is able to generate high-fidelity face images and shows more precise 3D controllability than state-of-the-art 2D-based controllable face synthesis methods.

## Requirements
 - We recommend Linux for performance and compatibility reasons.
 - 8 NVIDIA GPUs. We developed and trained the model using TITANXP.
 - We recommend running this repo on Docker. This project is built on top of Pi-GAN, Deep3DFaceRecon_pytorch and GFPGAN, which might take some time to build a compact environment. Emperically, we find the nvdiffrast environment especially hard to settle. So we follow the Dockerfile to build a compatible environment for our project.
```.bash
cd cgof/Deep3DFaceRecon_pytorch/nvdiffrast
docker build --tag cgof:latest -f ./Dockerfile . # build image
docker image ls # to find the IMAGE_ID
docker run -itd --gpus all  -v /home/kqsun:/home/kqsun -v /DATA:/DATA --workdir /home/kqsun/Tasks/eg3d --ipc=host --name cgof $IMAGE_ID /bin/bash # run image as contrainer
docker ps # to find the Contrainer_ID
docker exec -it $Contrainer_ID /bin/bash # enter container to work with your project
```

## Getting started
### Prepare pretrained models
To generate image with pretrained model, lease download the required [DATA](https://drive.google.com/drive/folders/1Oz66GAlpXFRcevCVEhHy-flXOID8jmyl?usp=sharing). Unzip it and put the folders to the corresponding path.
```.bash
mv DATA/Deep3DFaceRecon_pytorch/* cgof/Deep3DFaceRecon_pytorch/
mv DATA/GFPGAN_clean/* cgof/GFPGAN_clean/
mv DATA/outputs DATA/model cgof/
```

### Generating Contrllable Faces
control facial expressions
```.bash
CUDA_VISIBLE_DEVICES=0 python tools/eval/render/render_exp_ddp_verbose_gif.py outputs/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/generator.pth --output_dir imgs/exp/ --curriculum pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback  --image_size 128 --split False --save_depth False  --rt_norm --last_back
```

control head pose
```.bash
CUDA_VISIBLE_DEVICES=0 python tools/eval/render/render_pose_gif.py outputs/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/generator.pth --output_dir imgs/pose/ --curriculum pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback  --image_size 128  --rt_norm --last_back --seed 183
```

control identity
```.bash
CUDA_VISIBLE_DEVICES=0 python tools/eval/render/render_id_gif.py outputs/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/generator.pth --output_dir imgs/id/ --curriculum pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback  --image_size 128 --split False --save_depth False  --rt_norm --last_back
```

concatenate frames to a video
```.bash
python frames2video.py /home/kqsun/Tasks/cgof_release/cgof/imgs/pose/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/seed_183/splits/normals/183
```

### Training
```.bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train/train.py --output_dir ./output/ --load_dir ./output/ --curriculum xxx
```

## Citation
```
@article{sun2022controllable,
  title={Controllable 3d face synthesis with conditional generative occupancy fields},
  author={Sun, Keqiang and Wu, Shangzhe and Huang, Zhaoyang and Zhang, Ning and Wang, Quan and Li, HongSheng},
  journal={arXiv preprint arXiv:2206.08361},
  year={2022}
}
```

## Acknowledgement
This project is built on top of Pi-GAN, Deep3DFaceRecon_pytorch and GFPGAN. We thank the contributors of these prior projects for building such excellent code bases. We would like to thank Yu Deng for providing the evaluation code of the Disentanglement Score, and Jianzhu Guo for providing the pre-trained face reconstruction model. We are also indebted to thank Xingang Pan, Han Zhou, KwanYee Lin, Jingtan Piao, and Hang Zhou for their insightful discussions.
This work is supported in part by Centre for Perceptual and Interactive Intelligence Limited, in part by the General Research Fund through the Research Grants Council of Hong Kong under Grants.
