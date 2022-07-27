import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def stg_algorithm(grad, y, mean_var_list, model, input_size, max_iteration, device):
    grad = [g.to(device) for g in grad]
    model = model.to(device)
    y = y.to(device)
    dummy_x = torch.randn(input_size).to(device).requires_grad_(True)

    hook = BNStatisticsHook(model, train=True)
    optim = torch.optim.Adam([dummy_x], lr=0.1)
    scheduler = get_warmup_cosine_scheduler(optim, warm_up_iter=50, T_max=20000, lr_max=0.1, lr_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    for iteration in range(max_iteration):
        hook.clear()
        optim.zero_grad()
        model.zero_grad()

        dummy_loss = criterion(model(dummy_x), y)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
        dummy_mean_var_list = hook.mean_var_list

        grad_loss = 0
        for g1, g2 in zip(dummy_grad, grad):
            grad_loss += 0.001 * F.mse_loss(g1, g2, reduction="sum")

        bn_loss = 0
        for (m1, v1), (m2, v2) in zip(dummy_mean_var_list, mean_var_list):
            bn_loss += 0.1 * F.mse_loss(m1, m2, reduction="sum")
            bn_loss += 0.1 * F.mse_loss(v1, v2, reduction="sum")

        tv_loss = 1e-4 * tv_loss_fn(dummy_x)
        l2_loss = 1e-6 * l2_loss_fn(dummy_x)

        total_loss = grad_loss + bn_loss + tv_loss + l2_loss
        total_loss.backward()
        optim.step()
        scheduler.step()

        cur_lr = optim.state_dict()['param_groups'][0]['lr']
        with torch.no_grad():
            dummy_x += 0.2 * cur_lr * torch.randn(dummy_x.shape).to(device)
        print(
            f"\riter:{iteration}, "
            f"total:{total_loss:.8f}, "
            f"grad:{grad_loss:.8f}, "
            f"bn:{bn_loss:.8f}, "
            f"tv:{tv_loss:.8f}, "
            f"l2:{l2_loss:.8f}, "
            f"lr:{cur_lr:.8f}",
            end="")
    print("\nfinish gradient inversion!")
    hook.close()
    return dummy_x.detach()


def tv_loss_fn(x):
    bz = x.size(0)
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    # return (dx + dy) / bz
    return dx + dy


def l2_loss_fn(x):
    return torch.norm(x, p=2)


def get_warmup_cosine_scheduler(optim, warm_up_iter, T_max, lr_max, lr_min):
    lr_lambda = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else (lr_min + 0.5 * (
            lr_max - lr_min) * (1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


class BNStatisticsHook:
    def __init__(self, model, train=True):
        self.train = train
        self.mean_var_list = []
        self.hook_list = []
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                hook = module.register_forward_pre_hook(self.hook_fn)
                self.hook_list.append(hook)

    def hook_fn(self, _, input_data):
        mean = input_data[0].mean(dim=[0, 2, 3])
        var = input_data[0].var(dim=[0, 2, 3])
        if not self.train:
            mean = mean.detach().clone()
            var = var.detach().clone()
        self.mean_var_list.append([mean, var])

    def close(self):
        self.mean_var_list.clear()
        for hook in self.hook_list:
            hook.remove()

    def clear(self):
        self.mean_var_list.clear()
