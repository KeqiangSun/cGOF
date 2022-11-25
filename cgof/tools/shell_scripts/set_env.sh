nvcc --version
conda deactivate
conda remove -n pigan --all
conda create -n pigan python=3.8
conda activate pigan
pip install -r requirements_clean.txt
pip install pynvml
conda install IPython

# pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
# pip install -U git+https://github.com/fadel/pytorch_ema@3950a7b5c4b88f46fd14f620277bad21898597a9
git config --global user.email "469148505@qq.com"
git config --global user.name "KeqiangSun"

# pip install torch==1.8.0+cu101 torchvision==0.9.0+cu101 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=10.1 -c pytorch
# pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu101_pyt180/download.html
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
conda install Cython

cd tddfa
sh build.sh 
cd ..

gcc -shared -Wall -O3 render.c -o render.so -fPIC
=> gcc -shared -Wall -O3 render.c -std=c99 -o render.so -fPIC

# Data
scp /media/SSD/kqsun/data.tgz kqsun@uc163:/media/SSD/kqsun/
scp /home/kqsun/Tasks/models.tgz kqsun@uc163:/home/kqsun/Tasks/pigan/models/
scp /home/kqsun/Tasks/pigan/tddfa/configs/bfm_noneck_v3.pkl /home/kqsun/Tasks/pigan/tddfa/configs/param_mean_std_62d_120x120.pkl \
    tri.pkl kqsun@uc163:/home/kqsun/Tasks/pigan/tddfa/configs/
scp -r /home/kqsun/Tasks/pigan/outputs/CelebA_pigan_spade0_tddfa10_depth10_style10_pretrain2 \
    kqsun@uc163:/home/kqsun/Tasks/pigan/outputs/