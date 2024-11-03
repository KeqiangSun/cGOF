# Controllable 3D Face Synthesis with Conditional Generative Occupancy Fields (cGOF)
PyTorch implementation for the NeurIPS 2022 paper [cGOF](https://arxiv.org/abs/2206.08361) and the TPAMI 2023 paper [cGOF++](https://arxiv.org/abs/2211.13251).

<!-- ![Avatar](assets/teaser.png) -->

## Controllable 3D Face Synthesis with Conditional Generative Occupancy Fields
[Keqiang Sun](https://keqiangsun.github.io)*, [Shangzhe Wu](https://elliottwu.com)*, [Zhaoyang Huang](https://drinkingcoder.github.io), [Ning Zhang](https://scholar.google.com/citations?user=Hy0rk7IAAAAJ&hl=zh-TW), [Quan Wang](https://scholar.google.com/citations?user=KmxEHm4AAAAJ&hl=zh-TW), [Hongsheng Li](https://www.ee.cuhk.edu.hk/~hsli/) \
\* equal contribution

https://keqiangsun.github.io/projects/cgof/

Abstract: Capitalizing on the recent advances in image generation models, existing controllable face image synthesis methods are able to generate high-fidelity images with some levels of controllability, e.g., controlling the shapes, expressions, textures, and poses of the generated face images. However, these methods focus on 2D image generative models, which are prone to producing inconsistent face images under large expression and pose changes. In this paper, we propose a new NeRF-based conditional 3D face synthesis framework, which enables 3D controllability over the generated face images by imposing explicit 3D conditions from 3D face priors. At its core is a conditional Generative Occupancy Field (cGOF) that effectively enforces the shape of the generated face to commit to a given 3D Morphable Model (3DMM) mesh. To achieve accurate control over fine-grained 3D face shapes of the synthesized image, we additionally incorporate a 3D landmark loss as well as a volume warping loss into our synthesis algorithm. Experiments validate the effectiveness of the proposed method, which is able to generate high-fidelity face images and shows more precise 3D controllability than state-of-the-art 2D-based controllable face synthesis methods.

## Citation
```
@article{sun2022cgof,
  title={Controllable 3d face synthesis with conditional generative occupancy fields},
  author={Sun, Keqiang and Wu, Shangzhe and Huang, Zhaoyang and Zhang, Ning and Wang, Quan and Li, HongSheng},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={16331--16343},
  year={2022}
}
@article{sun2023cgof++,
  title={Cgof++: Controllable 3d face synthesis with conditional generative occupancy fields},
  author={Sun, Keqiang and Wu, Shangzhe and Zhang, Ning and Huang, Zhaoyang and Wang, Quan and Li, Hongsheng},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2023},
  publisher={IEEE}
}
```

## Acknowledgement
This project is built on top of Pi-GAN, EG3D, Deep3DFaceRecon_pytorch and GFPGAN. We thank the contributors of these prior projects for building such excellent code bases. We would like to thank Eric Ryan Chan for sharing the inversion script, Yu Deng for providing the evaluation code of the Disentanglement Score, and Jianzhu Guo for providing the pre-trained face reconstruction model. We are also indebted to thank Xingang Pan, Han Zhou, KwanYee Lin, Jingtan Piao, and Hang Zhou for their insightful discussions.
This work is supported in part by Centre for Perceptual and Interactive Intelligence Limited, in part by the General Research Fund through the Research Grants Council of Hong Kong under Grants.
