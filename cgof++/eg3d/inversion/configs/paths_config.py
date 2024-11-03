## Pretrained models paths
eg3d_ffhq = '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/inversion/inversion_utils/warp20/network-snapshot-001008.pkl'
# eg3d_ffhq = '/mnt/afs/kqsun/Tasks/eg3d/eg3d/inversion/inversion_utils/warp30/network-snapshot-001612.pkl'
# eg3d_ffhq = '/mnt/afs/kqsun/Tasks/eg3d/eg3d/inversion/inversion_utils/final_2200.pkl'
dlib = '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/inversion/inversion_utils/align.dat'

## Dirs for output files
checkpoints_dir = '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/inversion/outputs_0808/checkpoints_new'
embedding_base_dir = '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/inversion/outputs_0808/embeddings_new'
experiments_output_dir = '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/inversion/outputs_0808/output_new'
logdir = '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/inversion/outputs_0808/logs_new'

## Input info
# Location of the cameras json file
input_pose_path = '/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/proj_data/input3/epoch_20_000000/cameras.json'
# The image tag to lookup in the cameras json file
input_id = 'bieber'
# Where the input image resides
input_data_path = f'/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/proj_data/input3/crop_1024/{input_id}'
# input_data_path = f'/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d_pti_inversion-main/data/scene_1/ref_img/crop_1024/'
# Where the outputs are saved (i.e. embeddings/{input_data_id})
input_data_id = input_id
z_path = f'/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/proj_data/input2/crop_1024/epoch_20_000000/{input_id}.mat'
# z_path = f'/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d_pti_inversion-main/data/scene_1/ref_img/crop_1024/epoch_20_000000/{input_id}.mat'

## Keywords
pti_results_keyword = 'PTI'
e4e_results_keyword = 'e4e'
sg2_results_keyword = 'SG2'
sg2_plus_results_keyword = 'SG2_plus'
multi_id_model_type = 'multi_id'

# python edit_latent_code.py --network ./inversion/outputs/embeddings/anne/PTI/anne/model_anne.pt --w_path ./inversion/outputs/embeddings/anne/PTI/anne/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/anne.mat --outdir ./edit_results/anne/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100
# python edit_latent_code.py --network ./inversion/outputs/embeddings/biden/PTI/biden/model_biden.pt --w_path ./inversion/outputs/embeddings/biden/PTI/biden/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/biden.mat --outdir ./edit_results/biden/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100
# python edit_latent_code.py --network ./inversion/outputs/embeddings/blake/PTI/blake/model_blake.pt --w_path ./inversion/outputs/embeddings/blake/PTI/blake/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/blake.mat --outdir ./edit_results/blake/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100
# python edit_latent_code.py --network ./inversion/outputs/embeddings/lily/PTI/lily/model_lily.pt --w_path ./inversion/outputs/embeddings/lily/PTI/lily/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/lily.mat --outdir ./edit_results/lily/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100
# python edit_latent_code.py --network ./inversion/outputs/embeddings/obama/PTI/obama/model_obama.pt --w_path ./inversion/outputs/embeddings/obama/PTI/obama/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/obama.mat --outdir ./edit_results/obama/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100
# python edit_latent_code.py --network ./inversion/outputs/embeddings/tom/PTI/tom/model_tom.pt --w_path ./inversion/outputs/embeddings/tom/PTI/tom/optimized_noise_dict.pickle --c_path ./proj_data/input2/epoch_20_000000/cameras.json --z_path ./proj_data/input2/crop_1024/epoch_20_000000/tom.mat --outdir ./edit_results/tom/warp30_1612 --sample_mult 2 --use_face_recon True --trunc 1.0 --trunc-cutoff 100

