import torch
import torch.nn as nn
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from utils import show_imgs


def ig_algorithm(grad, x, y, model: nn.Module, input_size, max_iteration, device, record_dir):
    grad = [g.to(device) for g in grad]
    model = model.to(device)
    y = y.to(device)
    dummy_x = torch.rand(input_size).to(device).requires_grad_(True)
    optim = torch.optim.Adam([dummy_x], lr=0.1)
    criterion = nn.CrossEntropyLoss()

    # 3/8 5/8 7/8
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,
                                                     milestones=[max_iteration // 2.667,
                                                                 max_iteration // 1.6,
                                                                 max_iteration // 1.142], gamma=0.1)

    for iteration in range(max_iteration):
        optim.zero_grad()
        model.zero_grad()

        dummy_loss = criterion(model(dummy_x), y)
        dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        pnorm = [0, 0]
        costs = 0
        for g1, g2 in zip(dummy_grad, grad):
            costs -= (g1 * g2).sum()
            pnorm[0] += g1.pow(2).sum()
            pnorm[1] += g2.pow(2).sum()
        costs = 1 + costs / pnorm[0].sqrt() / pnorm[1].sqrt()

        tv_loss = 1e-1 * total_variation(dummy_x)

        total_loss = costs + tv_loss
        total_loss.backward()
        optim.step()
        scheduler.step()

        cur_lr = optim.state_dict()['param_groups'][0]['lr']

        if iteration % 10 == 0:
            psnr = peak_signal_noise_ratio(dummy_x, x, dim=0, data_range=1.0)
            print(
                f"\riter:{iteration}, "
                f"psnr:{psnr}, "
                f"total:{total_loss:.8f}, "
                f"costs:{costs:.8f}, "
                f"tv:{tv_loss:.8f}, "
                f"lr:{cur_lr:.8f}",
                end="")
    print("\nfinish gradient inversion!")
    return dummy_x.detach()


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
