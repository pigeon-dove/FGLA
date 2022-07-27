import torch
import argparse
from models import Resnet50
from utils import make_dir_if_not_exist, setup_seed
from gen_attack import train_generator, Generator

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", help="experiment name used to create folder", default="exp_0")
    parser.add_argument("--batch_size", help="batch size for training", default=24, type=int)
    parser.add_argument("--epochs", help="epochs for training", default=30, type=int)
    parser.add_argument("--device", help="which device to use", default="cuda:0")
    parser.add_argument("--seed", help="random seeds for experiments", default=1225, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    record_dir = f"./data/train_generator/{args.exp_name}/"
    make_dir_if_not_exist(record_dir)

    setup_seed(args.seed)
    resnet50 = Resnet50(pool=False)
    generator = Generator()

    train_generator(record_dir,
                    args.batch_size,
                    args.epochs,
                    generator,
                    resnet50,
                    args.device)

    print(f"finish training, see result in {record_dir}")
