import os
import cv2
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str)
opt = parser.parse_args()
input_dir = opt.path
if input_dir.endswith('/'):
    input_dir = input_dir[:-1]

img_array = []
# names = glob.glob('vids_aug_sr/restored_faces/*.png')
names = glob.glob(os.path.join(input_dir, '*.png'))
names.sort()
for filename in names:
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


# out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

# _cap = cv2.VideoCapture(0)
_fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output_path = input_dir + ".mp4"
out = cv2.VideoWriter(output_path, _fourcc, 15.0, size)


for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
