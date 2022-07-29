# %%
import matplotlib
import matplotlib.pyplot as plt

import torch
from models import Resnet50
from utils import setup_seed
from dataset import get_grad_dl, get_dataloader
from gen_attack import Generator
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from utils import show_imgs

from torchvision import transforms, datasets

# %%

imgs = torch.load("./data/reconstruct/exp_our_nv_2/dummy_x_1.pth")
for i, img in enumerate(imgs):
    img = img.permute([1, 2, 0])
    img = img.clamp(0, 1)
    img = img.detach().cpu().numpy()
    matplotlib.image.imsave(f"./test_{i}.png", img)

# %%
setup_seed(1225)
get_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

resnet50 = Resnet50(pool=False)
dataloader = get_dataloader("imagenet", batch_size=8, shuffle=True, train=False)
grad_dl = get_grad_dl(resnet50, dataloader, device="cuda:0")
dl = iter(grad_dl)

for x, y, grad, mean_var_list in dl:
    if len(y.unique()) != 8:
        for i, img in enumerate(x):
            img = img.permute([1, 2, 0])
            img = img.clamp(0, 1)
            img = img.detach().cpu().numpy()
            matplotlib.image.imsave(f"./test_true_{i}.png", img)
        break
transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor()]
)
dataset = datasets.Caltech256("./data/dataset/", transform=transform, download=False)

# %%
batch_size = 8
device = "cuda:1"
decoder = Generator()
decoder.load_state_dict(torch.load("./data/gen_weights.pth"))
decoder.eval()
setup_seed(1227)
get_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

resnet50 = Resnet50(pool=False)
dataloader = get_dataloader("imagenet", batch_size=batch_size, shuffle=True, train=False)
grad_dl = get_grad_dl(resnet50, dataloader, device="cpu")
dl = iter(grad_dl)
x, y, grad, mean_var_list = next(dl)

# zip_percent = 0.001
#
# num1 = int(grad[-2].numel() * zip_percent)
# threshold1 = grad[-2].abs().flatten().sort(descending=True)[0][num1]
#
# num2 = int(grad[-1].numel() * zip_percent)
# threshold2 = grad[-1].abs().flatten().sort(descending=True)[0][num2]
#

noise = 0
g_w = grad[-2] + torch.normal(0, noise, size=grad[-2].shape)
g_b = grad[-1] + torch.normal(0, noise, size=grad[-1].shape)

# g_w[g_w.abs() < threshold1] = 0.
# g_b[g_b.abs() < threshold2] = 0.

bz = len(y)
offset_w = torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(dim=0) * (bz - 1) / bz
offset_b = torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean() * (bz - 1) / bz
conv_out = (g_w[y] - offset_w) / (g_b[y] - offset_b).unsqueeze(1)
# conv_out = g_w[y] / g_b[y].unsqueeze(1)

conv_out[torch.isnan(conv_out)] = 0.
conv_out[torch.isinf(conv_out)] = 0.

conv_out, decoder = conv_out.to(device), decoder.to(device)
dummy_x = decoder(conv_out).detach().cpu()
x = x.detach().cpu()

show_imgs(dummy_x[0:1], x[0:1], save=False, dir_path="", filename="")

matplotlib.image.imsave(f"./{noise}.png", dummy_x[0:1].squeeze().permute([1, 2, 0]).numpy())
# matplotlib.image.imsave(f"./x.png", x.squeeze().permute([1, 2, 0]).numpy())

psnr = peak_signal_noise_ratio(dummy_x, x, dim=0, data_range=1.0)
ssim = structural_similarity_index_measure(dummy_x, x, data_range=1.0)
dummy_x_temp, x_temp = (dummy_x * 2 - 1).clamp(-1, 1), (x * 2 - 1).clamp(-1, 1)
lpips = get_lpips(dummy_x_temp, x_temp)

print(f"{psnr}\t{ssim}\t{lpips}")

# %%
batch_size = 512
device = "cuda:0"
decoder = GivNet()
decoder.eval()
decoder.load_state_dict(torch.load("./data/weights_25.pth"))
setup_seed(1234)
get_lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg")

resnet50 = Resnet50(pool=False)
dataloader = get_dataloader("imagenet", batch_size=batch_size, shuffle=True, train=False)
grad_dl = get_grad_dl(resnet50, dataloader, device="cpu")
dl = iter(grad_dl)

num = 0

for x, y, grad, mean_var_list in dl:
    if num >= 512:
        break
    g_w = grad[-2]
    g_b = grad[-1]

    bz = len(y)
    offset_w = torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(dim=0) * (bz - 1) / bz
    offset_b = torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean() * (bz - 1) / bz
    conv_out = (g_w[y] - offset_w) / (g_b[y] - offset_b).unsqueeze(1)

    conv_out[torch.isnan(conv_out)] = 0.
    conv_out[torch.isinf(conv_out)] = 0.

    conv_out, decoder = conv_out.to(device), decoder.to(device)

    psnr = 0
    ssim = 0
    lpips = 0
    j = 0

    sm_bz = 32
    if batch_size <= 32:
        sm_bz = batch_size
    for i in range(batch_size // sm_bz):
        j += 1
        c = decoder(conv_out[i:i + 32]).detach().cpu()
        t = x[i:i + 32].detach().cpu()

        psnr += peak_signal_noise_ratio(c, t, dim=0, data_range=1.0)
        ssim += structural_similarity_index_measure(c, t, data_range=1.0)
        dummy_x_temp, x_temp = (c * 2 - 1).clamp(-1, 1), (t * 2 - 1).clamp(-1, 1)
        lpips += get_lpips(dummy_x_temp, x_temp)

    num += batch_size
    print(f"{psnr / j}\t{ssim / j}\t{lpips / j}")

# %%
# dummy_x = dummy_x.squeeze().permute([1, 2, 0]).clamp(0, 1)

# plt.imshow(x_t)
# plt.show()
# plt.imshow(dummy_x)
# plt.show()
for idx, img in enumerate(dummy_x):
    img = img.permute([1, 2, 0])
    matplotlib.image.imsave(f"./{idx}.png", img.numpy())

# %%
import matplotlib.pyplot as plt

psnr_list = [18.56206003, 18.51245975, 18.37811732, 18.59561276, 18.848382, 18.98789883, 19.04623604]

ssim_list = [0.467589093, 0.474656705, 0.478203487, 0.487759497, 0.482781552, 0.491560221, 0.488615274]

lpips = [0.587845698, 0.580251627, 0.576109771, 0.570592396, 0.577967301, 0.578535974, 0.588060439]

font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 12
         }
plt.style.use("bmh")
fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_ylim(0, 25)
ax1.set_xlabel("Batch size")
ax1.set_ylabel("PSNR")
ax1.set_xticks(range(7))
ax1.set_xticklabels([8, 16, 32, 64, 128, 256, 512])
line1, = ax1.plot(range(7), psnr_list, "y--", label="PSNR")

ax2 = ax1.twinx()

ax2.set_ylim(0, 1)
ax2.set_ylabel("SSIM / LPIPS")
ax2.set_xticks(range(7))
ax2.set_xticklabels([8, 16, 32, 64, 128, 256, 512])
line2, = ax2.plot(range(7), ssim_list, "b-.", label="SSIM")

line3, = ax2.plot(range(7), lpips, "r-.", label="SSIM")

plt.legend([line1, line2, line3], ["PSNR", "SSIM", "LPIPS"])
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("bmh")

size = 3
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_ylim(0, 22)
ax1.set_ylabel("PSNR")

ax2 = ax1.twinx()

ax2.set_ylim(0, 1)
ax2.set_ylabel("SSIM / LPIPS")

x = np.arange(size)
psnr = [18.562, 21.24828997, 18.59516914]
ssim = [0.476, 0.795686131, 0.533742841]
lpips = [0.587, 0.434604486, 0.567923266]

total_width, n = 0.8, 3
width = total_width / n
x = x - (total_width - width) / 2

ax1.bar(x, psnr, width, color="r", label='a')
ax2.bar(x + width, ssim, width, color="g", label='b')
ax2.bar(x + 2 * width, lpips, width, color="b", label='c')
plt.legend()
plt.show()

# %%
import matplotlib.pyplot as plt

plt.style.use("bmh")

plt.figure(figsize=[8, 6])

class1000_offset = [1.55E-06, 3.31E-06, 1.05E-05, 4.78E-05, 1.94E-04]
class512_offset = [6.93E-06, 1.64E-05, 4.73E-05, 2.08E-04, 8.59E-04]
class256_offset = [2.51E-05, 5.76E-05, 1.91E-04, 8.19E-04, 3.57E-03]

class1000 = [8.29E-06, 3.30E-05, 1.50E-04, 7.18E-04, 3.02E-03]
class512 = [3.35E-05, 1.39E-04, 6.28E-04, 3.07E-03, 1.42E-02]
class256 = [1.34E-04, 5.53E-04, 2.68E-03, 1.43E-02, 8.21E-02]

plt.plot(range(5), class1000_offset, "o-", color="#D95319", markerfacecolor='white', label="FSG*, 1000 class")
plt.plot(range(5), class512_offset, "v--", color="#D95319", label="FSG*, 512 class")
plt.plot(range(5), class256_offset, "x:", color="#D95319", label="FSG*, 256 class")

plt.plot(range(5), class1000, "o-", color="#77AC30", markerfacecolor='white', label="FSG, 1000 class")
plt.plot(range(5), class512, "v--", color="#77AC30", label="FSG, 512 class")
plt.plot(range(5), class256, "x:", color="#77AC30", label="FSG, 256 class")

plt.yscale('log')
plt.xticks(range(5), [4, 8, 16, 32, 64])

plt.ylabel("MSE")
plt.xlabel("batch size")

plt.legend(facecolor='white')

plt.show()
