CUDA_VISIBLE_DEVICES=0 python vis.py --network ./outputs/eg3d_128/00003-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-004800.pkl --trunc 1.0 --outdir ./vis_results/128/eg3d/ --shapes False --reload_modules True --factor 1 --subject 33 --variation 5 --get_norm True --get_input True --angle_multiplier 0.0 --merge True --nrow 5  --neural_rendering_resolution 256 &
CUDA_VISIBLE_DEVICES=1 python vis.py --network ./outputs/eg3d_128_iter4800_recon4/00000-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-005200.pkl --trunc 1.0 --outdir ./vis_results/128/eg3d_recon4/ --shapes False --reload_modules True --factor 1 --subject 33 --variation 5 --get_norm True --get_input True --angle_multiplier 0.0 --merge True --nrow 5  --neural_rendering_resolution 256 &
CUDA_VISIBLE_DEVICES=2 python vis.py --network ./outputs/eg3d_128_iter4800_recon4_snm/00003-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-001800.pkl --trunc 1.0 --outdir ./vis_results/128/eg3d_recon4_snm/ --shapes False --reload_modules True --factor 1 --subject 33 --variation 5 --get_norm True --get_input True --angle_multiplier 0.0 --merge True --nrow 5  --neural_rendering_resolution 256 &
CUDA_VISIBLE_DEVICES=3 python vis.py --network ./outputs/eg3d_128_iter4800_recon4_snm_depr100/00001-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-000600.pkl --trunc 1.0 --outdir ./vis_results/128/eg3d_recon4_snm_depr100/ --shapes False --reload_modules True --factor 1 --subject 33 --variation 5 --get_norm True --get_input True --angle_multiplier 0.0 --merge True --nrow 5  --neural_rendering_resolution 256 &
CUDA_VISIBLE_DEVICES=4 python vis.py --network ./outputs/eg3d_128_iter4800_recon4_snm_depr100_ldmk6/00000-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-000800.pkl --trunc 1.0 --outdir ./vis_results/128/eg3d_recon4_snm_depr100_ldmk6 --shapes False --reload_modules True --factor 1 --subject 33 --variation 5 --get_norm True --get_input True --angle_multiplier 0.0 --merge True --nrow 5  --neural_rendering_resolution 256 &
CUDA_VISIBLE_DEVICES=5 python vis.py --network ./outputs/eg3d_128_iter4800_recon4_snm_depr100_ldmk6_warp30/00007-ffhq-FFHQ_128-gpus8-batch32-gamma1/network-snapshot-000800.pkl --trunc 1.0 --outdir ./vis_results/128/eg3d_recon4_snm_depr100_ldmk6_warp30 --shapes False --reload_modules True --factor 1 --subject 33 --variation 5 --get_norm True --get_input True --angle_multiplier 0.0 --merge True --nrow 5  --neural_rendering_resolution 256 &


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