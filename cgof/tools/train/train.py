"""Train pi-GAN. Supports distributed training."""
from packages import *
from pigan_trainer import PiganTrainer


def setup(rank, world_size, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_ADDR'] = port
    os.environ['MASTER_PORT'] = port
    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def train(rank, world_size, opt):
    torch.manual_seed(0)
    setup(rank, world_size, opt.port)
    torch.cuda.set_device(rank)
    trainer = PiganTrainer(rank, world_size, opt)
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs",
                        type=int,
                        default=3000,
                        help="number of epochs of training")
    parser.add_argument("--sample_interval",
                        type=int,
                        default=200,
                        help="interval between image sampling")
    parser.add_argument('--output_dir', type=str, default='debug')
    parser.add_argument('--load_dir', type=str, default='')
    parser.add_argument('--curriculum', type=str, required=True)
    parser.add_argument('--eval_freq', type=int, default=5000)
    parser.add_argument('--port', type=str, default='22479')
    parser.add_argument('--set_step', type=int, default=None)
    parser.add_argument('--model_save_interval', type=int, default=5000)
    parser.add_argument('--fid_output_num', type=int, default=2048)
    parser.add_argument('--save_depth',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--print_level', type=str, default='info')
    parser.add_argument('--fid',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=False)
    return parser.parse_args()
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/train/train.py --output_dir outputs/pigan_recon4_snm_depr10000_norm10 --load_dir outputs/pigan_recon4 --curriculum pigan_recon4_snm_depr10000_norm10 --save_depth

if __name__ == '__main__':
    opt = parse_args()
    pprint(opt)
    os.makedirs(opt.output_dir, exist_ok=True)
    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    # logger('trig trainging.')

    if num_gpus > 1:
        mp.spawn(train, args=(num_gpus, opt), nprocs=num_gpus, join=True)
    else:
        train(0, num_gpus, opt)

