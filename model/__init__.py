import torch.nn as nn
import torch
import torchvision
from utils import hubconf  # Changed from relative import to absolute import

from .resnet import BasicBlock, Bottleneck, resnet18, resnet50  # noqa: F401
from .regnet import ResBottleneckBlock, regnetx_600m, regnetx_3200m  # noqa: F401
from .mobilenetv2 import InvertedResidual, mobilenetv2  # noqa: F401
from .mnasnet import _InvertedResidual, mnasnet  # noqa: F401
from quant.quant_module import QuantizedLayer, QuantizedBlock, Quantizer   # noqa: F401

import timm
from timm.models.regnet import Bottleneck as RegNetBottleneck
from timm.models.vision_transformer import Block as VITBlock


class QuantBasicBlock(QuantizedBlock):
    """
    Implementation of Quantized BasicBlock used in ResNet-18 and ResNet-34.
    """
    def __init__(self, org_module: BasicBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput
        
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, config)
        self.conv2 = QuantizedLayer(org_module.conv2, None, config, qoutput=False)
        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, config, qoutput=False)
        self.activation = org_module.relu2
        
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)
            
            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None
            
            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3
            
            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1_relu(x)
        x = self.conv2(x)
        x += residual
        x = self.activation(x)
        
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


class QuantBottleneck(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Block used in ResNet-50, -101 and -152.
    """
    def __init__(self, org_module: Bottleneck, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput
        
        self.conv1_relu = QuantizedLayer(org_module.conv1, org_module.relu1, config)
        self.conv2_relu = QuantizedLayer(org_module.conv2, org_module.relu2, config)
        self.conv3 = QuantizedLayer(org_module.conv3, None, config, qoutput=False)

        if org_module.downsample is None:
            self.downsample = None
        else:
            self.downsample = QuantizedLayer(org_module.downsample[0], None, config, qoutput=False)
        self.activation = org_module.relu3
        
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1_relu(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x += residual
        x = self.activation(x)
        
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


class QuantResBottleneckBlock(QuantizedBlock):
    """
    Implementation of Quantized Bottleneck Blockused in RegNetX (no SE module).
    """
    def __init__(self, org_module: ResBottleneckBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.qoutput = qoutput
        self.out_mode = out_mode
        
        self.conv1_relu = QuantizedLayer(org_module.f.a, org_module.f.a_relu, config)
        self.conv2_relu = QuantizedLayer(org_module.f.b, org_module.f.b_relu, config)
        self.conv3 = QuantizedLayer(org_module.f.c, None, config, qoutput=False)
        if org_module.proj_block:
            self.downsample = QuantizedLayer(org_module.proj, None, config, qoutput=False)
        else:
            self.downsample = None
        self.activation = org_module.relu
        
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1_relu(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x += residual
        x = self.activation(x)
        
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


class QuantInvertedResidual(QuantizedBlock):
    """
    Implementation of Quantized Inverted Residual Block used in MobileNetV2.
    Inverted Residual does not have activation function.
    """
    def __init__(self, org_module: InvertedResidual, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.qoutput = qoutput
        self.out_mode = out_mode
        
        self.use_res_connect = org_module.use_res_connect
        if org_module.expand_ratio == 1:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.conv[0], org_module.conv[2], config),
                QuantizedLayer(org_module.conv[3], None, config, qoutput=False),
            )
        else:
            self.conv = nn.Sequential(
                QuantizedLayer(org_module.conv[0], org_module.conv[2], config),
                QuantizedLayer(org_module.conv[3], org_module.conv[5], config),
                QuantizedLayer(org_module.conv[6], None, config, qoutput=False),
            )
        
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p
        

    def forward(self, x):
        if self.use_res_connect:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


class _QuantInvertedResidual(QuantizedBlock):
    # mnasnet
    def __init__(self, org_module: _InvertedResidual, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.qoutput = qoutput
        self.out_mode = out_mode
        
        self.apply_residual = org_module.apply_residual
        self.conv = nn.Sequential(
            QuantizedLayer(org_module.layers[0], org_module.layers[2], config),
            QuantizedLayer(org_module.layers[3], org_module.layers[5], config),
            QuantizedLayer(org_module.layers[6], None, config, qoutput=False),
        )
        
        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(None, config.quant.a_qconfig_low)
            self.block_post_act_fake_quantize_med = Quantizer(None, config.quant.a_qconfig_med)
            self.block_post_act_fake_quantize_high = Quantizer(None, config.quant.a_qconfig_high)

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        if self.apply_residual:
            x = x + self.conv(x)
        else:
            x = self.conv(x)
        
        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x


def get_parent_and_name(root, full_name):
    names = full_name.split(".")
    parent = root
    for name in names[:-1]:
        parent = getattr(parent, name)
    return parent, names[-1]

# For ViT (ViT and DeiT)
class QuantVITBlock(QuantizedBlock):
    """
    Implementation of Quantized Vision Transformer Block used in Vision Transformer.
    """

    def __init__(self, orig_module: VITBlock, config, qoutput=True, out_mode="calib"):
        super().__init__()
        self.out_mode = out_mode
        self.qoutput = qoutput

        self.norm1 = orig_module.norm1

        # quantize Linear layers in Attention
        attn_linear = []
        self.attn = orig_module.attn
        # for name, module in self.attn.named_modules():
        #     if isinstance(module, nn.Linear):
        #         attn_linear.append((name, module))
        # for name, module in attn_linear:
        #     parent_module, last_name = get_parent_and_name(
        #         self.attn, name
        #     )  # Helper function to get parent module
        #     setattr(
        #         parent_module,
        #         last_name,
        #         QuantizedLayer(module, None, config, qoutput=False),
        #     )
        self.attn.qkv = QuantizedLayer(orig_module.attn.qkv, None, config, qoutput=False)
        self.attn.proj = QuantizedLayer(orig_module.attn.proj, None, config, qoutput=False)

        self.ls1 = orig_module.ls1
        self.drop_path1 = orig_module.drop_path1

        self.norm2 = orig_module.norm2

        # self.mlp = nn.Sequential(
        #     QuantizedLayer(orig_module.mlp.fc1, orig_module.mlp.act, config),
        #     orig_module.mlp.drop1,
        #     orig_module.mlp.norm,
        #     QuantizedLayer(orig_module.mlp.fc2, None, config, qoutput=False),
        #     orig_module.mlp.drop2,
        # )
        # quantize Linear layers in MLP block
        mlp_linear = []
        self.mlp = orig_module.mlp
        # for name, module in self.mlp.named_modules():
        #     if isinstance(module, nn.Linear):
        #         mlp_linear.append((name, module))
        # for name, module in mlp_linear:
        #     parent_module, last_name = get_parent_and_name(
        #         self.mlp, name
        #     )  # Helper function to get parent module
        #     if "fc2" in name:
        #         setattr(parent_module, last_name, QuantizedLayer(module, None, config, qoutput=False))  # Replace Linear layer
        #     elif "fc1" in name:
        #         setattr(parent_module, last_name, QuantizedLayer(module, None, config))
        self.mlp.fc1 = QuantizedLayer(orig_module.mlp.fc1, orig_module.mlp.act, config)
        self.mlp.fc2 = QuantizedLayer(orig_module.mlp.fc2, None, config, qoutput=False)
        self.mlp.act = nn.Identity()
        
        self.ls2 = orig_module.ls2
        self.drop_path2 = orig_module.drop_path2

        if self.qoutput:
            self.block_post_act_fake_quantize_low = Quantizer(
                None, config.quant.a_qconfig_low
            )
            self.block_post_act_fake_quantize_med = Quantizer(
                None, config.quant.a_qconfig_med
            )
            self.block_post_act_fake_quantize_high = Quantizer(
                None, config.quant.a_qconfig_high
            )

            self.f_l = None
            self.f_m = None
            self.f_h = None
            self.f_lmh = None

            self.lambda1 = config.quant.ptmq.lambda1
            self.lambda2 = config.quant.ptmq.lambda2
            self.lambda3 = config.quant.ptmq.lambda3

            self.mixed_p = config.quant.ptmq.mixed_p

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))

        if self.qoutput:
            if self.out_mode == "calib":
                self.f_l = self.block_post_act_fake_quantize_low(x)
                self.f_m = self.block_post_act_fake_quantize_med(x)
                self.f_h = self.block_post_act_fake_quantize_high(x)

                self.f_lmh = (
                    self.lambda1 * self.f_l
                    + self.lambda2 * self.f_m
                    + self.lambda3 * self.f_h
                )
                f_mixed = torch.where(torch.rand_like(x) < self.mixed_p, x, self.f_lmh)

                x = f_mixed
            elif self.out_mode == "low":
                x = self.block_post_act_fake_quantize_low(x)
            elif self.out_mode == "med":
                x = self.block_post_act_fake_quantize_med(x)
            elif self.out_mode == "high":
                x = self.block_post_act_fake_quantize_high(x)
            else:
                raise ValueError(
                    f"Invalid out_mode '{self.out_mode}': only ['low', 'med', 'high'] are supported"
                )
        return x

quant_modules = {
    BasicBlock: QuantBasicBlock,
    Bottleneck: QuantBottleneck,
    ResBottleneckBlock: QuantResBottleneckBlock,
    InvertedResidual: QuantInvertedResidual,
    _InvertedResidual: _QuantInvertedResidual,
    VITBlock: QuantVITBlock,
}


def load_model(config):
    config["kwargs"] = config.get("kwargs", dict())

    print(config["kwargs"])
    print(config["type"])

    # Extract model name from config and load using timm
    if config["type"].startswith(
        ("vit", "deit")
    ):  # Assuming the config type uses ViT or any timm model
        model = timm.create_model(config["type"], pretrained=True)
    else:
        model = hubconf.__dict__[config["type"]](pretrained=True, **config["kwargs"])  # Added line to load CNN models from hubconf

    return model


def set_qmodel_block_aqbit(model, out_mode):
    """
    Function to set the activation bitwidth for all quantized blocks during inference
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantizedBlock):
            # print(name)
            module.out_mode = out_mode

def set_qmodel_weight_qbit(model, out_mode):
    """
    Function to set the weight bitwidth for all quantized blocks during inference
    """
    for name, module in model.named_modules():
        if isinstance(module, QuantizedLayer):
            if hasattr(module, "out_mode"):
                module.out_mode = out_mode