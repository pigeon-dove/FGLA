import datetime
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from utils import show_imgs, make_dir_if_not_exist
from dataset import get_imagenet_exp_dl
from torch.utils.data import DataLoader

import os
import torch.distributed as dist


def train_generator(result_dir, batch_size, epochs, decoder, origin_model, device):
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")

    origin_model.fc = nn.Sequential()  # remove linear layer
    origin_model = origin_model.cuda(local_rank)

    decoder = decoder.cuda(local_rank)
    decoder = torch.nn.parallel.DistributedDataParallel(decoder, device_ids=[local_rank], output_device=local_rank)

    imagenet_dir = "./data/dataset/imagenet/"
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor()]
    )
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop([224, 224]),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor()]
    )
    train_ds = datasets.ImageNet(imagenet_dir, split="train", transform=train_transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    val_ds = datasets.ImageNet(imagenet_dir, split="val", transform=transform)
    train_dl = DataLoader(train_ds, batch_size, num_workers=4, pin_memory=True, shuffle=False, sampler=train_sampler)
    val_dl = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True, shuffle=False)

    criterion = nn.MSELoss()
    optim = torch.optim.Adam(decoder.parameters(), lr=0.01)
    for epoch in range(0, epochs):
        train_sampler.set_epoch(epoch)
        start = datetime.datetime.now()
        decoder.train()
        train_loss = 0
        for idx, (img, y) in enumerate(train_dl):
            img = img.cuda(local_rank)
            optim.zero_grad()
            pre_img = decoder(origin_model(img))
            loss = criterion(pre_img, img)
            loss.backward()
            optim.step()
            if idx % 10 == 0:
                show_training_state(idx, len(train_dl), loss.item(), True, datetime.datetime.now() - start)
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl)
        pre_img, img = pre_img.detach().cpu(), img.detach().cpu()
        show_imgs(pre_img, img, save=True, dir_path=f"{result_dir}/images/", filename=f"{epoch}_train")

        if local_rank == 0:
            decoder.eval()
            val_loss = 0
            with torch.no_grad():
                for idx, (img, y) in enumerate(val_dl):
                    img = img.cuda(local_rank)
                    pre_img = decoder(origin_model(img))
                    loss = criterion(pre_img, img)
                    if idx % 10 == 0:
                        show_training_state(idx, len(val_dl), loss.item(), False, datetime.datetime.now() - start)
                    val_loss += loss.item()
            val_loss = val_loss / len(val_dl)
            pre_img, img = pre_img.detach().cpu(), img.detach().cpu()
            show_imgs(pre_img, img, save=True, dir_path=f"{result_dir}/images/", filename=f"{epoch}_val")

            show_final_state(epoch, train_loss, val_loss, datetime.datetime.now() - start)
            save_history(epoch, train_loss, val_loss, result_dir=result_dir)
            save_weights(epoch, decoder.module, result_dir=result_dir)


def save_history(epoch, train_loss, val_loss, result_dir):
    """
    Save the losses from the training process to a csv file
    """
    make_dir_if_not_exist(result_dir)
    with open(f"{result_dir}/history.csv", "a") as f:
        f.write(f"{epoch}, {train_loss}, {val_loss}\n")


def show_training_state(idx, length, loss, train: bool, time):
    print("\r" + " " * 50, end="")
    status = "validating"
    if train:
        status = "training"
    print(f"\r{status} step:{idx}/{length} loss:{loss} {time}", end="")


def show_final_state(epoch, train_loss, val_loss, time):
    print("\r" + " " * 50, end="")
    print(f"\repoch:{epoch} train_loss:{train_loss} val_loss:{val_loss} {time}", end="\n")


def save_weights(epoch, model, result_dir):
    make_dir_if_not_exist(result_dir)
    torch.save(model.state_dict(), f"{result_dir}/weights_{epoch}.pth")
