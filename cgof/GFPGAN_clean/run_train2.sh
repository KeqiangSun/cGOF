CUDA_VISIBLE_DEVICES=0,1,2,3 BASICSR_JIT=True python -m torch.distributed.launch --nproc_per_node=4 --master_port=22024 gfpgan/train.py -opt options/test_v1_simple_ffhq_celebhq.yml --launcher pytorch --auto_resume