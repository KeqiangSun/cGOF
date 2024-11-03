import os
import numpy as np
import skimage.io
import skimage.transform
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# all attributes
all_attr = ['cloth', 'ear_r', 'eye_g', 'hat',
            'hair', 'l_brow', 'l_ear', 'l_eye',
            'l_lip', 'mouth', 'neck_l', 'neck',
            'nose', 'r_brow', 'r_ear', 'r_eye',
            'skin', 'u_lip']

# 416 henry cavill
# 417 chase from house
# 514 bryan cranston

bad_idx = [49, 64, 70, 94, 132, 151, 193, 210, 250, 252, 259, 330, 341, 344, 374, 418, 442, 465, 472, 486, 502, 515, 548, 594, 584, 636, 676, 680, 699, 709, 833, 864, 874, 879, 947, 973, 985, 999, 1036, 1064, 1102, 1174, 1178, 1205, 1236]
# mouth_idx = [0, 39, 45, 60, 159, 188, 189, 190, 239, 263, 329, 379, 435, 436, 452]  there are more...


# read attributes
def get_smiling_mask():
    with open('CelebAMask-HQ-attribute-anno.txt', 'r') as f:
        n_samples = int(f.readline().rstrip())
        attr = f.readline().rstrip().split()

        smiling_idx = np.argmax(['Smiling' in att for att in attr])

        smiling = []
        for i in range(n_samples):
            curr_attr = f.readline().rstrip().split()
            smiling.append(int(curr_attr[smiling_idx+1]))

    return smiling


def make_head_dataset():

    smiling_mask = get_smiling_mask()

    # remove neck and cloth
    attr = ['ear_r', 'eye_g', 'hat',
            'hair', 'l_brow', 'l_ear', 'l_eye',
            'l_lip',
            'nose', 'r_brow', 'r_ear', 'r_eye',
            'skin', 'u_lip']

    fnames = sorted(glob('./masks/*.png'))

    # group by image
    fnames_chunked = []
    for fname in tqdm(fnames):
        name = os.path.basename(fname)
        idx = int(name[:5])

        if len(fnames_chunked) < idx + 1:
            fnames_chunked.append([])

        fnames_chunked[idx].append(fname)

    def get_masked_img(idx, fnames):
        if idx in bad_idx:
            return

        if smiling_mask[idx] > 0:
            return

        if idx > (1400 + len(bad_idx) - 1):
            return

        rgb = skimage.io.imread(f'CelebA-HQ-img/{idx}.jpg').astype(np.float) / 255.

        fg_mask = np.zeros((512, 512, 3))
        head_mask = np.zeros((512, 512, 3))

        for fname in fnames:
            if any(att in fname for att in all_attr):
                fg_mask += skimage.io.imread(fname)
            else:
                print(fname)

            if any(att in fname for att in attr):
                head_mask += skimage.io.imread(fname)

        # plt.subplot(121)
        # plt.imshow(rgb)
        # plt.subplot(122)
        # plt.imshow(fg_mask)
        # plt.show()

        # remove mouth
        out_mouth = None
        for fname in fnames:
            if 'mouth' in fname:
                mouth_mask = skimage.io.imread(fname)

                assert mouth_mask.shape[-1] == 3, 'mouth mask is weird size'

                if np.sum(mouth_mask[..., 0] > 0) > 1000:
                    return
                else:
                    # mask -= mouth_mask
                    # mouth_mask = skimage.transform.resize(mouth_mask, (1024, 1024))
                    # out_mouth = (mouth_mask * 255).astype(np.uint8)
                    # skimage.io.imsave(f'masked/{idx:05d}_mouth.png', out_mouth, check_contrast=False)
                    pass

        # if out_mouth is None:
        #     skimage.io.imsave(f'masked/{idx:05d}_mouth.png',
        #                       np.zeros((1024, 1024, 3)).astype(np.uint8),
        #                       check_contrast=False)

        # apply foreground mask and save
        fg_mask = skimage.transform.resize(fg_mask, (1024, 1024))
        fg_mask[fg_mask > 0] = 1
        # out = ((rgb * fg_mask + (1 - fg_mask)) * 255).astype(np.uint8)
        out = (rgb * 255).astype(np.uint8)
        skimage.io.imsave(f'masked/{idx:05d}.png', out)

        # save out head mask
        head_mask = skimage.transform.resize(head_mask, (1024, 1024))
        head_mask[head_mask > 0] = 1
        head_mask = (head_mask * 255).astype(np.uint8)
        skimage.io.imsave(f'masked/{idx:05d}_mask.png', head_mask)

    Parallel(n_jobs=30)(delayed(get_masked_img)(idx, fnames) for idx, fnames in enumerate(tqdm(fnames_chunked)))

    # remove any over 500
    fnames = sorted(glob('masked/*.png'))

    if len(fnames) > 1000:
        for fname in fnames[1000:]:
            os.remove(fname)


def visualize_dataset(nrows=8, ncols=8):
    imgs = []
    for i in range(nrows*ncols):
        imgs.append(skimage.io.imread(f'masked/{i:05d}.png'))

    imgs = [imgs[i:i+ncols] for i in range(0, nrows*ncols, ncols)]

    imgs = [np.concatenate(img, axis=1) for img in imgs]
    imgs = np.concatenate(imgs, axis=0)

    imgs = skimage.transform.resize(imgs, (1024, 1024))
    skimage.io.imsave('imgs.png', imgs)


if __name__ == '__main__':
    make_head_dataset()
    # visualize_dataset()
