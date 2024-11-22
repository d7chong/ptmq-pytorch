import torch
import torch.nn as nn
import torch.nn.functional as F

# from torchvision.models.vision_transformer import MLPBlock, EncoderBlock
from quant.observer import (
    MSEFastObserver,
    MinMaxObserver,
    AvgMinMaxObserver,
    MSEObserver,
    AvgMSEObserver,
    AvgMSEFastObserver,
)
from quant.fake_quant import (
    AdaRoundFakeQuantize,
    FixedFakeQuantize,
    LSQFakeQuantize,
    LSQPlusFakeQuantize,
)


ObserverDict = {
    "MinMaxObserver": MinMaxObserver,
    "AvgMinMaxObserver": AvgMinMaxObserver,
    "MSEObserver": MSEObserver,
    "AvgMSEObserver": AvgMSEObserver,
    "MSEFastObserver": MSEFastObserver,
    "AvgMSEFastObserver": AvgMSEFastObserver,
}

FakeQuantizeDict = {
    "FixedFakeQuantize": FixedFakeQuantize,
    "LSQFakeQuantize": LSQFakeQuantize,
    "LSQPlusFakeQuantize": LSQPlusFakeQuantize,
    "AdaRoundFakeQuantize": AdaRoundFakeQuantize,
}


def ActivationQuantizer(a_qconfig):
    return FakeQuantizeDict[a_qconfig.quantizer](
        ObserverDict[a_qconfig.observer],
        bit=a_qconfig.bit,
        symmetric=a_qconfig.symmetric,
        ch_axis=a_qconfig.ch_axis,
    )


def WeightQuantizer(w_qconfig):
    return FakeQuantizeDict[w_qconfig.quantizer](
        ObserverDict[w_qconfig.observer],
        bit=w_qconfig.bit,
        symmetric=w_qconfig.symmetric,
        ch_axis=w_qconfig.ch_axis,
    )


class QuantizedOperator:
    pass


class QConv2d(QuantizedOperator, nn.Conv2d):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        padding_mode,
        w_qconfig,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return self._conv_forward(input, self.weight_fake_quant(self.weight), self.bias)


class QLinear(QuantizedOperator, nn.Linear):

    def __init__(self, in_features, out_features, bias, w_qconfig):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.linear(input, self.weight_fake_quant(self.weight), self.bias)


class QEmbedding(QuantizedOperator, nn.Embedding):

    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx,
        max_norm,
        norm_type,
        scale_grad_by_freq,
        sparse,
        _weight,
        w_qconfig,
    ):
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
        )
        self.weight_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.embedding(
            input,
            self.weight_fake_quant(self.weight),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


# Quantized Layer Norm
class QLayerNorm(QuantizedOperator, nn.LayerNorm):
    def __init__(self, normalized_shape, eps, elementwise_affine, w_qconfig):
        super().__init__(normalized_shape, eps, elementwise_affine)
        self.weight_fake_quant = WeightQuantizer(w_qconfig)
        self.bias_fake_quant = WeightQuantizer(w_qconfig)

    def forward(self, input):
        return F.layer_norm(
            input,
            self.normalized_shape,
            self.weight_fake_quant(self.weight),
            self.bias_fake_quant(self.bias),
        )


module_type_to_quant_weight = {
    nn.Linear: QLinear,
    nn.Conv2d: QConv2d,
    nn.Embedding: QEmbedding,
    nn.LayerNorm: QLayerNorm,
}


def get_module_args(module):
    if isinstance(module, nn.Linear):
        return dict(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
        )
    elif isinstance(module, nn.Conv2d):
        return dict(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
        )
    elif isinstance(module, nn.Embedding):
        return dict(
            num_embeddings=module.num_embeddings,
            embedding_dim=module.embedding_dim,
            padding_idx=module.padding_idx,
            max_norm=module.max_norm,
            norm_type=module.norm_type,
            scale_grad_by_freq=module.scale_grad_by_freq,
            sparse=module.sparse,
            _weight=None,
        )
    elif isinstance(module, nn.LayerNorm):
        return dict(
            normalized_shape=module.normalized_shape,
            eps=module.eps,
            elementwise_affine=module.elementwise_affine,
        )
    else:
        raise NotImplementedError


def Quantizer(module, config):
    if module is None:
        return ActivationQuantizer(a_qconfig=config)
    module_type = type(module)
    if module_type in module_type_to_quant_weight:
        kwargs = get_module_args(module)
        qmodule = module_type_to_quant_weight[module_type](
            **kwargs, w_qconfig=config.quant.w_qconfig
        )
        qmodule.weight.data = module.weight.data.clone()
        if getattr(module, "bias", None) is not None:
            qmodule.bias.data = module.bias.data.clone()
        return qmodule
    return module


class QuantizedModule(nn.Module):
    def __init__(self):
        super().__init__()


class QuantizedLayer(QuantizedModule):
    def __init__(self, module, activation, config, qoutput=True):
        super().__init__()
        self.qoutput = qoutput
        self.module = Quantizer(module, config)

        self.activation = activation
        if qoutput:
            self.layer_post_act_fake_quantize = Quantizer(
                None, config.quant.a_qconfig_low
            )

    def forward(self, x):
        x = self.module(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.qoutput:
            x = self.layer_post_act_fake_quantize(x)
        return x


class QuantizedBlock(QuantizedModule):
    def __init__(self):
        super().__init__()