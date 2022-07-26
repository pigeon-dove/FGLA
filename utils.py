import torch
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 300


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def show_imgs(imgs, true_imgs, save: bool, dir_path, filename):
    """
    Display of real and generated images, with the option to save as a file or not
    """
    imgs = imgs.clamp(0, 1)
    true_imgs = true_imgs.clamp(0, 1)
    imgs = imgs.permute([0, 2, 3, 1])
    true_imgs = true_imgs.permute([0, 2, 3, 1])
    img_num = len(imgs)
    plt.figure(figsize=[2, img_num])
    for idx, img in enumerate(true_imgs):
        plt.subplot(img_num, 2, 2 * idx + 1)
        plt.axis("off")
        plt.imshow(img)
    for idx, img in enumerate(imgs):
        plt.subplot(img_num, 2, 2 * idx + 2)
        plt.axis("off")
        plt.imshow(img)
    if save:
        make_dir_if_not_exist(dir_path)
        plt.savefig(f"{dir_path}/{filename}.png")
        plt.close()
    else:
        plt.show()
    plt.close()


def progress_bar(cur, total):
    total_num = 20
    progress_num = total_num * cur // total
    return progress_num * "#" + (total_num - progress_num) * "=" + f"{100*cur/total:3.1f}%"


def make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path, 0o0777)
