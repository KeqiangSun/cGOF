## Architechture
lpips_type = 'alex'
first_inv_type = 'w'
optim_type = 'adam'

# dataloading
batch_size = 4

## Locality regularization
latent_ball_num_of_samples = 1
locality_regularization_interval = 1
use_locality_regularization = False
use_noise_regularization = False 
use_mouth_inpainting = True
regulizer_l2_lambda = 0.1
regulizer_lpips_lambda = 0.1
regulizer_alpha = 30

## Loss
pt_l2_lambda = 1
pt_lpips_lambda = 1
pt_temporal_photo_lambda = 0
pt_temporal_depth_lambda = 0
temporal_consistency_loss = False

## Steps
LPIPS_value_threshold = 0.06
first_inv_steps = 450
max_pti_steps = 350
max_images_to_invert = 30

## Optimization
pti_learning_rate = 3e-4
first_inv_lr = 5e-3
train_batch_size = 1
use_last_w_pivots = False

