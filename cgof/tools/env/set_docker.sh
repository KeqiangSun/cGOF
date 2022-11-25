# Uninstall 
sudo service docker stop

sudo apt-get remove -y nvidia-docker2
sudo apt-get remove docker docker-engine docker.io containerd runc
sudo apt-get remove docker-ce docker-ce-cli containerd.io

sudo rm -rf /etc/systemd/system/docker.service.d
sudo rm -rf /var/lib/docker

# Install docker with gpu support
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
sudo docker run --rm --gpus all nvidia/cuda:10.2-base nvidia-smi

# Add user to docker group
sudo groupadd docker
sudo gpasswd -a kqsun docker
newgrp docker

# Build Image and Container
docker build --tag gltorch:latest -f docker_pigan/Dockerfile . # build image
docker run -itd --gpus all  -v /home/kqsun:/home/kqsun --workdir /home/kqsun/Tasks/pigan --ipc=host --name pigan gltorch /bin/bash
docker ps
docker exec -it cid /bin/bash
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train/train.py --curriculum pigan_recon4_snm_depr10000 --output_dir outputs/pigan_recon4_snm_depr10000 --load_dir outputs/pigan_recon4_ckpt --save_depth --print_level 'info' --port 22476