import torch
import torch.nn as nn


class StraightThrough(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


def _fold_bn(conv_module, bn_module):
    w_conv = conv_module.weight.data
    y_mean = bn_module.running_mean
    y_var = bn_module.running_var
    safe_std = torch.sqrt(y_var + bn_module.eps)
    w_view = (conv_module.out_channels, 1, 1, 1)

    if bn_module.affine:
        weight = w_conv * (bn_module.weight / safe_std).view(w_view)
        beta = bn_module.bias - bn_module.weight * y_mean / safe_std
        if conv_module.bias is not None:
            bias = bn_module.weight * conv_module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = w_conv / safe_std.view(w_view)
        beta = -y_mean / safe_std
        if conv_module.bias is not None:
            bias = conv_module.bias / safe_std + beta
        else:
            bias = beta
    return weight, bias


def fold_bn_into_conv(conv_module, bn_module):
    w, b = _fold_bn(conv_module, bn_module)

    if conv_module.bias is None:
        conv_module.bias = nn.Parameter(b)
    else:
        conv_module.bias.data = b
    conv_module.weight.data = w

    # set bn running stats
    bn_module.running_mean = bn_module.bias.data
    bn_module.running_var = bn_module.weight.data**2


def reset_bn(module: nn.BatchNorm2d):
    if module.track_running_stats:
        module.running_mean.zero_()
        module.running_var.fill_(1 - module.eps)
    if module.affine:
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def is_bn(m):
    return isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d)


def is_absorbing(m):
    return isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)


def search_fold_and_remove_bn(model):
    model.eval()
    prev = None
    for name, module in model.named_children():
        if is_bn(module) and is_absorbing(prev):
            fold_bn_into_conv(prev, module)
            setattr(model, name, StraightThrough())
        elif is_absorbing(module):
            prev = module
        else:
            prev = search_fold_and_remove_bn(module)
    return prev
