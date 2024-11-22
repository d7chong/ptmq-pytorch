import torch
import torch.nn as nn
import numpy as np

import time
import copy
import logging
import argparse
from tqdm import tqdm
import warnings

# disable wandb
# os.environ["WANDB_MODE"] = "disabled"

# Suppress the specific deprecation warning
warnings.filterwarnings("ignore", message="_aminmax is deprecated")


import utils
import utils.eval_utils as eval_utils
from utils.ptmq_recon import ptmq_reconstruction
from utils.fold_bn import search_fold_and_remove_bn, StraightThrough
from model import quant_modules, load_model, set_qmodel_block_aqbit
from quant.quant_state import (
    enable_calib_without_quant,
    enable_quantization,
    disable_all,
)
from quant.quant_module import QuantizedLayer, QuantizedBlock
from quant.fake_quant import QuantizeBase
from quant.observer import ObserverBase

logger = logging.getLogger("ptmq")
torch.set_float32_matmul_precision("high")


def quantize_model(model, config):
    def replace_module(module, config, qoutput=True):
        children = list(iter(module.named_children()))
        ptr, ptr_end = 0, len(children)
        prev_qmodule = None

        while ptr < ptr_end:
            tmp_qoutput = qoutput if ptr == ptr_end - 1 else True
            name, child_module = children[ptr][0], children[ptr][1]

            if (
                type(child_module) in quant_modules
            ):  # replace blocks with quantized blocks
                setattr(
                    module,
                    name,
                    quant_modules[type(child_module)](
                        child_module, config, tmp_qoutput
                    ),
                )
            elif isinstance(child_module, (nn.Conv2d, nn.Linear)):
                setattr(
                    module,
                    name,
                    QuantizedLayer(child_module, None, config, qoutput=tmp_qoutput),
                )
                prev_qmodule = getattr(module, name)
            elif isinstance(child_module, (nn.ReLU, nn.ReLU6, nn.GELU)):
                if prev_qmodule is not None:
                    prev_qmodule.activation = child_module
                    setattr(module, name, StraightThrough())
                else:
                    pass
            elif isinstance(child_module, StraightThrough):
                pass
            else:
                replace_module(child_module, config, tmp_qoutput)
            ptr += 1

    # we replace all layers to be quantized with quantization-ready layers
    replace_module(model, config, qoutput=False)

    # we store all modules in the quantized model (weight_module or activation_module)
    w_list, a_list = [], []
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if "weight" in name:
                w_list.append(module)
            elif "act" in name:
                a_list.append(module)

    # set first and last layer to 8-bit
    w_list[0].set_bit(8)
    w_list[-1].set_bit(8)

    # set the last layer's output to 8-bit
    a_list[-1].set_bit(8)

    logger.info(f"Finished quantizing model: {str(model)}")

    return model


def get_calib_data(train_loader, num_samples):
    calib_data = []
    for batch in train_loader:
        calib_data.append(batch[0])
        if len(calib_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(calib_data, dim=0)[:num_samples]


def main(config_path):
    # get config for applying ptmq
    config = eval_utils.parse_config(config_path)
    eval_utils.set_seed(config.process.seed)
    
    if args.model:
        config.model.type = args.model
    if args.w_bit:
        config.quant.w_qconfig.bit = args.w_bit
    if args.a_bit_low:
        config.quant.a_qconfig_low.bit = args.a_bit_low
    if args.a_bit_med:
        config.quant.a_qconfig_med.bit = args.a_bit_med
    if args.a_bit_high:
        config.quant.a_qconfig_high.bit = args.a_bit_high
    if args.scale_lr:
        config.quant.recon.scale_lr = args.scale_lr
    if args.recon_iter:
        config.quant.recon.iters = args.recon_iter
    
    print(f"Model: {config.model.type}")
    print(f"W{config.quant.w_qconfig.bit}A{config.quant.a_qconfig_low.bit}{config.quant.a_qconfig_med.bit}{config.quant.a_qconfig_high.bit}")
    print(f"Scale learning rate: {config.quant.recon.scale_lr}")
    print(f"Reconstruction iterations: {config.quant.recon.iters}")

    train_loader, val_loader = eval_utils.load_data(config, **config.data)
    calib_data = get_calib_data(train_loader, config.quant.calibrate).cuda()

    model = load_model(config.model)  # load original model

    # print(model)
    model.cuda()
    model = model.eval()
    
    search_fold_and_remove_bn(model)  # remove+fold batchnorm layers
   
    # quanitze model if config.quant is defined
    if hasattr(config, "quant"):
        model = quantize_model(model, config)

    model.cuda()  # move model to GPU
    model.eval()  # set model to evaluation mode

    fp_model = copy.deepcopy(model)  # save copy of full precision model
    disable_all(fp_model)  # disable all quantization

    # set names for all ObserverBase modules
    # ObserverBase modules are used to store intermediate values during calibration
    for name, module in model.named_modules():
        if isinstance(module, ObserverBase):
            module.set_name(name)

    print("Starting model calibration...")
    tik = time.time()

    model.eval()
    
    enable_calib_without_quant(model, quantizer_type="act_fake_quant")
    with torch.no_grad() and torch.inference_mode():
        model(calib_data[:256].cuda())

    # weight param calibration
    enable_calib_without_quant(model, quantizer_type="weight_fake_quant")
    with torch.no_grad() and torch.inference_mode():
        model(calib_data[:2].cuda())

    tok = time.time()

    logger.info(f"Calibration of {str(model)} took {tok - tik} seconds")
    print("Completed model calibration")

    print("Starting block reconstruction...")
    tik = time.time()
    # Block reconstruction (layer reconstruction for first & last layers)a
    if hasattr(config.quant, "recon"):
        enable_quantization(model)

        def recon_model(module, fp_module):
            for name, child_module in module.named_children():
                if isinstance(child_module, (QuantizedLayer, QuantizedBlock)):
                    ptmq_reconstruction(
                        model,
                        fp_model,
                        child_module,
                        name,
                        getattr(fp_module, name),
                        calib_data,
                        config.quant,
                    )
                else:
                    recon_model(child_module, getattr(fp_module, name))

        recon_model(model, fp_model)
    tok = time.time()
    print("Completed block reconstruction")
    print(f"PTMQ block reconstruction took {tok - tik:.2f} seconds")

    a_qmodes = ["low", "med", "high"]
    w_qbit = config.quant.w_qconfig.bit
    a_qbits = [
        config.quant.a_qconfig_low.bit,
        config.quant.a_qconfig_med.bit,
        config.quant.a_qconfig_high.bit,
    ]

    # save ptmq model
    torch.save(
        model.state_dict(), f"ptmq_w{w_qbit}_a{a_qbits[0]}{a_qbits[1]}{a_qbits[2]}.pth"
    )

    enable_quantization(model)

    for a_qmode, a_qbit in zip(a_qmodes, a_qbits):
        set_qmodel_block_aqbit(model, a_qmode)

        print(
            f"Starting model evaluation of W{w_qbit}A{a_qbit} block reconstruction ({a_qmode})..."
        )
        acc1, acc5 = eval_utils.validate_model(val_loader, model)

        print(f"Top-1 accuracy: {acc1:.2f}, Top-5 accuracy: {acc5:.2f}")
        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c", "--config", default="config/resnet18.yaml", type=str, help="Path to config file"
    )
    parser.add_argument(
        "-m", "--model", default="resnet18", type=str, help="Model to be quantized"
    )
    parser.add_argument(
        "-w", "--w_bit", default=8, type=int, help="Weight bitwidth for quantization"
    )
    parser.add_argument(
        "-al", "--a_bit_low", default=6, type=int, help="Activation bitwidth for quantization"
    )
    parser.add_argument(
        "-am", "--a_bit_med", default=7, type=int, help="Activation bitwidth for quantization"
    )
    parser.add_argument(
        "-ah", "--a_bit_high", default=8, type=int, help="Activation bitwidth for quantization"
    )
    parser.add_argument(
        "-lr", "--scale_lr", default=4e-5, type=float, help="Learning rate for scale"
    )
    parser.add_argument(
        "-i", "--recon_iter", default=100, type=int, help="Number of reconstruction iterations"
    )
    parser.add_argument(
        "-o", "--observer", default=None, type=str, help="Observer type for quantization"
    )
    args = parser.parse_args()

    main(args.config)