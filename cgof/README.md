<!-- ```console
# control facial expressions
python tools/eval/render/render_exp.py outputs/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/generator.pth --output_dir imgs/exp/ --curriculum pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback --seeds 31 --image_size 128 --split False --save_depth False --rows 1 --exp_num 12 --angle_multiplier 0
CUDA_VISIBLE_DEVICES=0 python tools/eval/render/render_pose_gif.py outputs/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/generator.pth --output_dir imgs/pose/ --curriculum pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback  --image_size 128  --rt_norm --last_back --seed 183
CUDA_VISIBLE_DEVICES=0 python tools/eval/render/render_exp_ddp_verbose_gif.py outputs/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/generator.pth --output_dir imgs/exp/ --curriculum pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback  --image_size 128 --split False --save_depth False  --rt_norm --last_back
CUDA_VISIBLE_DEVICES=0 python tools/eval/render/render_id_gif.py outputs/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/generator.pth --output_dir imgs/id/ --curriculum pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback  --image_size 128 --split False --save_depth False  --rt_norm --last_back
python frames2video.py /home/kqsun/Tasks/cgof_release/cgof/imgs/pose/pigan_recon4_snm_depr10000_norm1000_lm3d10_warp3d10_bgdepr10000_georeg500_lastback/seed_183/splits/normals/183
``` -->

# Controllable 3D Face Synthesis with Conditional Generative Occupancy Fields (cGOF)
### [Project Page](https://keqiangsun.github.io/projects/cgof/) | [Paper](https://arxiv.org/abs/2206.08361)
[Keqiang Sun](https://keqiangsun.github.io)\*,
[Shangzhe Wu](https://elliottwu.com)\*,
[Zhaoyang Huang](https://drinkingcoder.github.io),
[Ning Zhang](https://scholar.google.com/citations?user=Hy0rk7IAAAAJ&hl=zh-TW),
[Quan Wang](https://scholar.google.com/citations?user=KmxEHm4AAAAJ&hl=zh-TW),
[Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/)
<br>
\*denotes equal contribution

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
