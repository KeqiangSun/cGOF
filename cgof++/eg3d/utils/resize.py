import os
import numpy as np
from PIL import Image
import argparse


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--size', type=int, default=512, help='resize image to what size')
    opt = parser.parse_args()
    return opt


def resize_dir_pil(inp_dir, size=(128, 128)):
    h, w = size
    inp_dir = os.path.dirname(inp_dir+'/')
    out_dir = inp_dir + f'_{h}'
    for r, ds, fs in os.walk(inp_dir):
        for f in fs:
            img_path = os.path.join(r, f)
            img = Image.open(img_path)
            out = img.resize(size, Image.ANTIALIAS)
            save_path = img_path.replace(inp_dir, out_dir)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            out.save(save_path)


if __name__ == "__main__":
    args = parse()
    size = args.size
    resize_dir_pil(args.path, size=(size, size))
