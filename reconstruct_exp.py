import os
import torch
import argparse
import time
from models import Resnet50
from utils import make_dir_if_not_exist, setup_seed, progress_bar, show_imgs
from dataset import get_grad_dl, get_dataloader
from optim_attack import dlg_algorithm, stg_algorithm, ig_algorithm
from gen_attack import gen_attack_algorithm, Generator
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", help="choose fgla, dlg, stg, ig to use", default="fgla")
    parser.add_argument("--model_weights", help="weights path for generator", default="./data/weights_25.pth")
    parser.add_argument("--reconstruct_num", help="number of reconstructed batches", default=20, type=int)
    parser.add_argument("--dataset", help="dataset used to reconstruct", default="imagenet")
    parser.add_argument("--max_iteration", help="iteration to reconstruct", default=20_000, type=int)
    parser.add_argument("--exp_name", help="experiment name used to create folder", default="exp_0")
    parser.add_argument("--batch_size", help="batch size for training", default=8, type=int)
    parser.add_argument("--device", help="which device to use", default="cuda:1")
    parser.add_argument("--seed", help="random seeds for experiments", default=1234, type=int)
    return parser.parse_args()


def save_history(index, psnr, ssim, lpips, time, dir_path):
    with open(f"{dir_path}/history.csv", "a") as f:
        f.write(f"{index}, {psnr}, {ssim}, {lpips}, {time}\n")


if __name__ == "__main__":
    args = parse_args()
    record_dir = f"./data/reconstruct/{args.exp_name}/"
    if os.path.exists(record_dir):
        raise Exception(f"dir {record_dir} exits! use another exp_name")
    if args.algorithm not in ["fgla", "dlg", "stg", "ig"]:
        raise Exception(f"can not find algorithm {args.algorithm}!")
    if args.dataset not in ["imagenet", "cifar100", "caltech256"]:
        raise Exception(f"can not find dataset {args.dataset}!")

    make_dir_if_not_exist(record_dir)
    setup_seed(args.seed)
    get_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

    resnet50 = Resnet50(pool=False)
    dataloader = get_dataloader(args.dataset, batch_size=args.batch_size, shuffle=True, train=False)
    grad_dl = get_grad_dl(resnet50, dataloader, device=args.device)

    index = 0
    for x, y, grad, mean_var_list in grad_dl:
        start = time.time()
        if len(y.unique()) != args.batch_size:
            continue

        index += 1

        if args.algorithm == "dlg":
            dummy_x = dlg_algorithm(grad, y, resnet50, (args.batch_size, 3, 224, 224), args.max_iteration, args.device)
        elif args.algorithm == "stg":
            dummy_x = stg_algorithm(grad, y, mean_var_list, resnet50, (args.batch_size, 3, 224, 224),
                                    args.max_iteration,
                                    args.device)
        elif args.algorithm == "ig":
            dummy_x = ig_algorithm(grad, x, y, resnet50, (args.batch_size, 3, 224, 224), args.max_iteration,
                                   args.device, record_dir)
        else:
            decoder = Generator()
            decoder.load_state_dict(torch.load(args.model_weights))
            dummy_x = gen_attack_algorithm(grad, y, decoder, True, args.device)

        dummy_x, x = dummy_x.cpu(), x.cpu()
        psnr = peak_signal_noise_ratio(dummy_x, x, dim=0, data_range=1.0)
        ssim = structural_similarity_index_measure(dummy_x, x, data_range=1.0)
        dummy_x_temp, x_temp = (dummy_x * 2 - 1).clamp(-1, 1), (x * 2 - 1).clamp(-1, 1)
        lpips = get_lpips(dummy_x_temp, x_temp)

        print(f"idx:{index} "
              f"psnr:{psnr} "
              f"ssim:{ssim} "
              f"lpips:{lpips}")
        save_history(index, psnr, ssim, lpips, time.time() - start, f"{record_dir}")
        torch.save(dummy_x, f"{record_dir}/dummy_x_{index}.pth")
        show_imgs(dummy_x, x, save=True, dir_path=f"{record_dir}/images/", filename=f"{index}")
        if index >= args.reconstruct_num:
            break

    print("\nexp finish!")
