CUDA_VISIBLE_DEVICES=1 python tools/eval/eval_pigan_.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10/generator.pth --range 0 100 1 --curriculum pigan_recon4_snm_depr10000_norm1000_lm10 --save_depth --output_dir eval/pigan_recon4_snm_depr10000_norm1000_lm10 --using_cross_test --using_correlation_test --snm
CUDA_VISIBLE_DEVICES=1 python tools/eval/eval_pigan_.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10/generator.pth --range 0 100 1 --curriculum pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10 --save_depth --output_dir eval/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10/ --using_cross_test --using_correlation_test --snm
CUDA_VISIBLE_DEVICES=1 python tools/eval/eval_pigan_.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500/generator.pth --range 0 100 1 --curriculum pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500 --save_depth --output_dir eval/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500/ --using_cross_test --using_correlation_test --snm
CUDA_VISIBLE_DEVICES=1 python tools/eval/eval_pigan_.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback_ckpt/generator.pth --range 0 100 1 --curriculum pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback --save_depth --output_dir eval/pigan_recon4_snm_depr10000_norm1000_lm10_expwarp10_bgdepr10000_georeg500_lastback/ --using_cross_test --using_correlation_test --snm
CUDA_VISIBLE_DEVICES=1 python tools/eval/eval_pigan_.py outputs/pigan_recon4_snm_depr10000_norm1000_lm10_warp3d10_bgdepr10000_georeg500_lastback_lm3d300_ckpt/generator.pth --range 0 100 1 --curriculum pigan_recon4_snm_depr10000_norm1000_lm10_warp3d10_bgdepr10000_georeg500_lastback_lm3d300 --save_depth --output_dir eval/pigan_recon4_snm_depr10000_norm1000_lm10_warp3d10_bgdepr10000_georeg500_lastback_lm3d300 --using_cross_test --using_correlation_test --snm
CUDA_VISIBLE_DEVICES=1 python tools/eval/eval_pigan_.py outputs/pigan_recon4_lm10_expwarp10_lastback/generator.pth --range 0 100 1 --curriculum pigan_recon4_lm10_expwarp10_lastback --save_depth --output_dir eval/pigan_recon4_lm10_expwarp10_lastback --using_cross_test --using_correlation_test