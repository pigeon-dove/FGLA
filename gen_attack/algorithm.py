import torch


def gen_attack_algorithm(grad, y, generator, offset: bool, device):
    conv_out = flg(grad, y, offset)
    conv_out, decoder = conv_out.to(device), generator.to(device)
    imgs = generator(conv_out).detach().cpu()
    return imgs


def flg(grad, y, offset: bool):
    bz = len(y)
    g_w = grad[-2]
    g_b = grad[-1]
    if offset:
        offset_w = torch.stack([g for idx, g in enumerate(g_w) if idx not in y], dim=0).mean(dim=0) * (bz - 1) / bz
        offset_b = torch.stack([g for idx, g in enumerate(g_b) if idx not in y], dim=0).mean() * (bz - 1) / bz
        conv_out = (g_w[y] - offset_w) / (g_b[y] - offset_b).unsqueeze(1)
    else:
        conv_out = g_w[y] / g_b[y]
    conv_out[torch.isnan(conv_out)] = 0.
    conv_out[torch.isinf(conv_out)] = 0.
    return conv_out
