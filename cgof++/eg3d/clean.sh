# 对人脸属性进行变化
# cd /home/kqsun/Tasks/eg3d_022/eg3d/eg3d/
CUDA_VISIBLE_DEVICES=0 python vis.py \
--network /mnt/afs/kqsun/Tasks/eg3d/eg3d/outputs/network-snapshot-001612.pkl \
--trunc 0.7 \
--outdir ./vis_results/eg3d_recon4_snm_depr100_ldmk6_warp30 \
--shapes False \
--reload_modules True \
--factor 1 \
--subject 33 \
--variation 25 \
--get_norm False \
--get_input True \
--angle_multiplier 0.0 \
--merge True \
--nrow 5  \
--neural_rendering_resolution 256

# 使用Deep3DFaceRecon_pytorch进行人脸检测、对齐
# cd /mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d_pti_inversion-main/Deep3DFaceRecon_pytorch
python process_test_images.py --input_dir /mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d_pti_inversion-main/data/scene_1/ref_img --gpu 0

# 使用GAN Inversion获得给定图像的z
# 改：/home/kqsun/Tasks/eg3d_022/eg3d/eg3d/inversion/configs/paths_config.py
# cd /mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d
python inversion/run_pti.py

# 编辑Inversion的结果
# cd /home/kqsun/Tasks/eg3d_022/eg3d/eg3d/
python edit_latent_code.py --network ./inversion/outputs/embeddings/tom/PTI/tom/model_tom.pt --w_path ./inversion/outputs/embeddings/tom/PTI/tom/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/tom.mat --outdir ./edit_results/tom/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100

# 编辑Inversion的结果（设置不同camera pose）
# cd /home/kqsun/Tasks/eg3d_022/eg3d/eg3d/
python edit_latent_code.py --network ./inversion/outputs/embeddings/tom/PTI/tom/model_tom.pt --w_path ./inversion/outputs/embeddings/tom/PTI/tom/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/tom.mat --outdir ./edit_results/tom/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100

