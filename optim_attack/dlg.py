import torch
import torch.nn as nn
import torch.nn.functional as F


def dlg_algorithm(grad, y, model: nn.Module, input_size, max_iteration, device):
    grad = [g.to(device) for g in grad]
    model = model.to(device)
    y = y.to(device)
    dummy_x = torch.randn(input_size).to(device).requires_grad_(True)
    optim = torch.optim.LBFGS([dummy_x], lr=1)
    criterion = nn.CrossEntropyLoss()

    for iteration in range(max_iteration):
        def closure():
            optim.zero_grad()
            model.zero_grad()

            dummy_loss = criterion(model(dummy_x), y)
            dummy_grad = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

            grad_loss = 0
            for g1, g2 in zip(dummy_grad, grad):
                grad_loss += F.mse_loss(g1, g2, reduction="sum")

            grad_loss.backward()
            return grad_loss

        grad_loss = optim.step(closure)
        print(
            f"\riter:{iteration}, "
            f"loss:{grad_loss:.8f}",
            end="")
    print("\nfinish dlg!")
    return dummy_x.detach()
