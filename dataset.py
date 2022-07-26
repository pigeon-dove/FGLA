import copy
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from hook import BNStatisticsHook

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()]
)


def get_imagenet_exp_dl(batch_size):
    imagenet_dir = "./data/dataset/imagenet/"
    train_ds = datasets.ImageNet(imagenet_dir, split="train", transform=transform)
    val_ds = datasets.ImageNet(imagenet_dir, split="val", transform=transform)
    train_dl = DataLoader(train_ds, batch_size, num_workers=8, pin_memory=True, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size, num_workers=8, pin_memory=True, shuffle=False)
    return train_dl, val_dl


def get_dataloader(name, batch_size, shuffle=False, train=False):
    if name == "imagenet":
        split = "train" if train else "val"
        dataset = datasets.ImageNet("./data/dataset/imagenet/", split=split, transform=transform)
    elif name == "caltech256":
        dataset = datasets.Caltech256("./data/dataset/", transform=transform, download=True)
    elif name == "cifar100":
        dataset = datasets.CIFAR100("./data/dataset/cifar100", transform=transform, download=True, train=train)
    return DataLoader(dataset, batch_size, drop_last=True, shuffle=shuffle)


def get_grad_dl(model: nn.Module, dataloader: DataLoader, device):
    model = copy.deepcopy(model).to(device)
    criterion = nn.CrossEntropyLoss()
    hook = BNStatisticsHook(model, train=False)
    for x, y in dataloader:
        model.zero_grad()
        hook.clear()
        x, y = x.to(device), y.to(device)
        # y = torch.arange(len(x)).to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
        grad = torch.autograd.grad(loss, model.parameters())
        grad = [g.detach() for g in grad]
        mean_var_list = hook.mean_var_list
        yield x, y, grad, mean_var_list
