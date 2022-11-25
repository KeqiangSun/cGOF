BASICSR_JIT=True python -m torch.distributed.launch --nproc_per_node=4 --master_port=22021 gfpgan/train.py -opt options/test_v1_simple.yml --launcher pytorch --auto_resume
