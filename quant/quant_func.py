import torch


def round_ste(x):
    """
    Rounds the input tensor to the nearest integer value.
    """
    return (x.round() - x).detach() + x


def grad_scale(x, scale):
    """
    Scales the input tensor by a factor of `scale`.
    """
    return (x - (x * scale)).detach() + (x * scale)


def fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max):
    """
    Default per-tensor quantization function.
    """
    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_per_channel_affine(
    x, scale, zero_point, ch_axis, quant_min, quant_max
):
    """
    Default per-channel quantization function.
    """
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]

    scale = scale.reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)

    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_tensor_affine_train(
    x, scale, zero_point, quant_min, quant_max, grad_factor
):
    """
    Per-tensor quantization function for LSQ.
    """
    scale = grad_scale(scale, grad_factor)

    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnable_per_channel_affine_train(
    x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor
):
    """
    Per-channel quantization function for LSQ.
    """
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]

    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = zero_point.reshape(new_shape)

    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_tensor_affine_train(
    x, scale, zero_point, quant_min, quant_max, grad_factor
):
    """
    Per-tensor quantization function for LSQ+.
    """
    zero_point = (zero_point.round() - zero_point).detach() + zero_point

    scale = grad_factor(scale, grad_factor)
    zero_point = grad_scale(zero_point, grad_factor)

    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant


def fake_quantize_learnableplus_per_channel_affine_train(
    x, scale, zero_point, ch_axis, quant_min, quant_max, grad_factor
):
    """
    Per-channel quantization function for LSQ+.
    """
    zero_point = (zero_point.round() - zero_point).detach() + zero_point
    new_shape = [1] * len(x.shape)
    new_shape[ch_axis] = x.shape[ch_axis]

    scale = grad_scale(scale, grad_factor).reshape(new_shape)
    zero_point = grad_scale(zero_point, grad_factor).reshape(new_shape)

    x_int = round_ste(x / scale) + zero_point
    x_quant = torch.clamp(x_int, quant_min, quant_max)
    x_dequant = (x_quant - zero_point) * scale
    return x_dequant
