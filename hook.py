import torch.nn as nn


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
