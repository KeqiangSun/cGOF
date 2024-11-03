import sys
# sys.path.append("../")
sys.path.append("/mnt/afs/kqsun/Tasks/eg3d_022/eg3d/eg3d/")

from torch.utils.tensorboard import SummaryWriter
from random import choice
from string import ascii_uppercase
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os
from configs import global_config, paths_config, hyperparameters
import shutil

from pti_training.coaches.single_id_coach import SingleIDCoach
from pti_training.coaches.single_id_coach_grayscale import SingleIDCoachGrayscale
from inversion_utils.ImagesDataset import ImagesDataset, GrayscaleImagesDataset, DECADataset
#from utils.parse_args import parse_args


def run_PTI(run_name='', use_wandb=False, reload_modules=False, use_face_recon=False):
    #parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    dataset = ImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        transforms.ToTensor()]))
    """
    dataset = GrayscaleImagesDataset(paths_config.input_data_path, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])]))
    """
    dataloader = DataLoader(dataset, batch_size=hyperparameters.batch_size, shuffle=False)

    coach = SingleIDCoach(dataloader, use_wandb, reload_modules, use_face_recon)
    #coach = SingleIDCoachGrayscale(dataloader, use_wandb)

    coach.train()
    return global_config.run_name


if __name__ == '__main__':
    run_PTI(
        run_name='',
        reload_modules=True,
        use_face_recon=True
    )
