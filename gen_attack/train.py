import datetime
import torch
import torch.nn as nn
from utils import show_imgs, make_dir_if_not_exist
from dataset import get_imagenet_exp_dl


def train_generator(result_dir, batch_size, epochs, generator, origin_model, device):
    origin_model.fc = nn.Sequential()  # remove linear layer
    origin_model = origin_model.to(device)

    generator = generator.to(device)
    criterion = nn.MSELoss()
    train_dl, val_dl = get_imagenet_exp_dl(batch_size)
    optim = torch.optim.Adam(generator.parameters(), lr=0.0001)

    for epoch in range(0, epochs):
        start = datetime.datetime.now()
        generator.train()
        train_loss = 0
        for idx, (img, y) in enumerate(train_dl):
            img = img.to(device)
            optim.zero_grad()
            pre_img = generator(origin_model(img))
            loss = criterion(pre_img, img)
            loss.backward()
            optim.step()
            if idx % 10 == 0:
                show_training_state(idx, len(train_dl), loss.item(), True, datetime.datetime.now() - start)
            train_loss += loss.item()
        train_loss = train_loss / len(train_dl)
        pre_img, img = pre_img.detach().cpu(), img.detach().cpu()
        show_imgs(pre_img, img, save=True, dir_path=f"{result_dir}/images/", filename=f"{epoch}_train")

        generator.eval()
        val_loss = 0
        with torch.no_grad():
            for idx, (img, y) in enumerate(val_dl):
                img = img.to(device)
                pre_img = generator(origin_model(img))
                loss = criterion(pre_img, img)
                if idx % 10 == 0:
                    show_training_state(idx, len(val_dl), loss.item(), False, datetime.datetime.now() - start)
                val_loss += loss.item()
        val_loss = val_loss / len(val_dl)
        pre_img, img = pre_img.detach().cpu(), img.detach().cpu()
        show_imgs(pre_img, img, save=True, dir_path=f"{result_dir}/images/", filename=f"{epoch}_val")

        show_final_state(epoch, train_loss, val_loss, datetime.datetime.now() - start)
        save_history(epoch, train_loss, val_loss, result_dir=result_dir)
        save_weights(epoch, generator, result_dir=result_dir)


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
