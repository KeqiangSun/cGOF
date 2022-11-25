#!/bin/sh
 
gpu_num=8                               # 可用gpu数
job_name=pigan                          # 任务名
partition=ha_face                       # 分区名
tcp=tcp://10.5.30                       # TCPIP通信端口IP前缀
port=23725                              # TCPIP通信端口编号
slurm=SH-IDC1-10-5-30                   # 使用的集群节点IP前缀
node=197                                # 使用的集群节点编号
currenttime=$(date +"%Y%m%d_%H%M%S")    # 使用时间戳作为不同log文件的区分
 
GLOG_logtostderr=1 MV2_USE_GPUDIRECT_GDRCOPY=0 MV2_ENABLE_AFFINITY=0 MV2_SMP_USE_CMA=0 MV2_USE_CUDA=1 \
srun  -p ${partition} -w ${slurm}-${node} --mpi=pmi2 \
        --gres=gpu:${gpu_num} -n1 --ntasks-per-node=${gpu_num} \    # 一个卡一个进程，因此该节点的进程数即为该节点可用的gpu数
        --job-name=${job_name} \
python -m torch.distributed.launch --nproc_per_node=${gpu_num} \
        --master_port=${port} train_slurm.py \
        --curriculum CelebAwoAMP \
        --output_dir outputs/CelebAwoAMPOutputDir \
        --init_method="${tcp}.${node}:${port}" \                    # tcp://10.5.30.197:23725
        --gpus=${gpu_num} \
        & 2>&1 | tee train_log/train_${currenttime}.log


echo "Done"