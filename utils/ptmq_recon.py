import numpy as np
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import wandb
import timm

from utils.eval_utils import DataSaverHook, StopForwardException, parse_config
from quant.quant_module import QuantizedModule, QuantizedBlock
from quant.fake_quant import LSQFakeQuantize, LSQPlusFakeQuantize, QuantizeBase, AdaRoundFakeQuantize

logger = logging.getLogger("ptmq")

# CONFIG_PATH = 'config/gpu_config.yaml'
# cfg = parse_config(CONFIG_PATH)

wandb.init(
    # set the wandb project where this run will be logged
    project="ptmq-pytorch",
    # track hyperparameters and run metadata
    config={
        "architecture": "ViT-S16",
        "dataset": "ImageNet1K-val",
        # "recon_iters": cfg.quant.recon.iters,
    },
)


def save_inp_oup_data(
    model,
    module,
    calib_data: list,
    store_inp=False,
    store_oup=False,
    bs: int = 32,
    keep_gpu: bool = True,
):
    """_summary_

    Args:
        model (nn.Module): model to be used for calibration
        module (nn.Module): quantized module
        calib_data (list): calibration data
        store_inp (bool): whether to store input data
        store_oup (bool): whether to store output data
        bs (int): batch size
        keep_gpu (bool): whether to store data on GPU
    """
    device = next(model.parameters()).device
    data_saver = DataSaverHook(
        store_input=store_inp, store_output=store_oup, stop_forward=True
    )
    handle = module.register_forward_hook(data_saver)
    cached = [[], []]
    with torch.no_grad() and torch.inference_mode():
        for i in range(int(calib_data.size(0) / bs)):
            try:
                _ = model(calib_data[i * bs : (i + 1) * bs].to(device))
            except StopForwardException:
                pass
            if store_inp:
                if keep_gpu:
                    cached[0].append(data_saver.input_store[0].detach())
                else:
                    cached[0].append(data_saver.input_store[0].detach().cpu())
            if store_oup:
                if keep_gpu:
                    cached[1].append(data_saver.output_store.detach())
                else:
                    cached[1].append(data_saver.output_store.detach().cpu())
    if store_inp:
        cached[0] = torch.cat([x for x in cached[0]])
    if store_oup:
        cached[1] = torch.cat([x for x in cached[1]])
    handle.remove()
    torch.cuda.empty_cache()
    return cached


class LinearTempDecay:
    def __init__(self, t_max=20000, warm_up=0.2, start_b=20, end_b=2):
        self.t_max = t_max
        self.start_decay = warm_up * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """_summary_

        Cosine annealing scheduler for temperature b.

        Args:
            t_max (int, optional): maximum number of iterations
            warm_up (float, optional): warm-up ratio
            start_b (int, optional): starting temperature
            end_b (int, optional): ending temperature

        Returns:
            temperature b
        """
        if t < self.start_decay:
            return self.start_b
        elif t > self.t_max:
            return self.end_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))


def gd_loss(f_fp, f_l, f_m, f_h, f_lmh, qconfig):
    gamma1 = qconfig.ptmq.gamma1
    gamma2 = qconfig.ptmq.gamma2
    gamma3 = qconfig.ptmq.gamma3

    loss_fp = torch.nn.functional.mse_loss(f_fp, f_lmh, reduction="mean")
    loss_hm = torch.nn.functional.mse_loss(f_h, f_m, reduction="mean")
    loss_hl = torch.nn.functional.mse_loss(f_h, f_l, reduction="mean")
    gd_loss = gamma1 * loss_fp + gamma2 * loss_hm + gamma3 * loss_hl

    # wandb.log(
    #     {
    #         'MSE(f_fp, f_lmh)': loss_fp,
    #         'MSE(f_h, f_m)': loss_hm,
    #         'MSE(f_h, f_l)': loss_hl
    #     }
    # )

    return gd_loss


class LossFunction:
    """
    loss = reconstrcution_loss + round_loss
    - recon_loss -> gd_loss from PTMQ
    - round_loss -> AdaRound loss
    """

    def __init__(
        self,
        module: QuantizedModule,
        weight: float = 1.0,
        iters=20000,
        b_range=(20, 2),
        warm_up=0.0,
        p=2.0,
        qconfig=None,
        use_gd_loss=False,
    ):
        self.module = module
        self.weight = weight
        self.loss_start = iters * warm_up
        self.p = p

        self.temp_decay = LinearTempDecay(
            iters, warm_up=warm_up, start_b=b_range[0], end_b=b_range[1]
        )
        self.qconfig = qconfig
        self.use_gd_loss = use_gd_loss

        """
        """
        # TEMP
        self.recon_loss = None
        self.round_loss = None

        self.count = 0

    def __call__(self, fp_block_output, q_block_output, f_l, f_m, f_h, f_lmh):
        """
        Compute the total loss for adaptive rounding with ptmq
        - total_loss = recon_loss + round_loss
            - recon_loss: GD loss (between f_fp, f_h, f_m, f_l, f_mixed)
            - round_loss: regularization term to optimize the rounding policy (AdaRound)
        """
        self.count += 1

        # Compute reconstruction loss
        recon_loss = 0.0
        if self.use_gd_loss:
            recon_loss = gd_loss(fp_block_output, f_l, f_m, f_h, f_lmh, self.qconfig)

        # Compute rounding loss
        b = self.temp_decay(self.count)
        if self.count < self.loss_start:
            round_loss = 0.0
        else:
            round_loss = 0.0
            for layer in self.module.modules():
                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    round_vals = layer.weight_fake_quant.rectified_sigmoid()
                    round_loss += (
                        self.weight * (1 - ((round_vals - 0.5).abs() * 2).pow(b)).sum()
                    )

        # Get total loss
        total_loss = recon_loss + round_loss
        """
        """
        # TEMP
        self.recon_loss = recon_loss
        self.round_loss = round_loss

        # Print loss
        # if self.count % 500 == 0:
        #     logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
        #         float(total_loss.item()), float(recon_loss.item()), float(round_loss), b, self.count))
        if isinstance(total_loss, float):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            total_loss = torch.tensor(total_loss, device=device, requires_grad=True)

        return total_loss


def ptmq_reconstruction(
    q_model, fp_model, q_module, name, fp_module, calib_data, qconfig
):
    # get model name as a string from the model
    model_name = fp_model.__class__.__name__
    wandb.run.name = f"{model_name}_lr{qconfig.recon.scale_lr}_iters{qconfig.recon.iters}"
    
    device = next(q_module.parameters()).device

    # Get data first (save input/output for each batch of calibration data through the blocks)
    q_block_inputs, _ = save_inp_oup_data(
        q_model,
        q_module,
        calib_data,
        store_inp=True,
        store_oup=False,
        bs=qconfig.recon.batch_size,
        keep_gpu=qconfig.recon.keep_gpu,
    )
    fp_block_inputs, _ = save_inp_oup_data(
        fp_model,
        fp_module,
        calib_data,
        store_inp=True,
        store_oup=False,
        bs=qconfig.recon.batch_size,
        keep_gpu=qconfig.recon.keep_gpu,
    )

    # Store quantization parameters for both weights and activations
    w_para, a_para = [], []
    for name, q_layer in q_module.named_modules():
        # collect layer weight quantization params
        if isinstance(q_layer, (nn.Linear, nn.Conv2d)):
            # print(f"w_para from: {name}")
            weight_quantizer = q_layer.weight_fake_quant
            weight_quantizer.init(q_layer.weight.data, qconfig.recon.round_mode)
            w_para += [weight_quantizer.alpha]
        # collect activation quantization params
        if isinstance(q_layer, QuantizeBase) and "post_act_fake_quantize" in name:
            # layer.drop_prob = qconfig.recon.drop_prob
            # print(f"a_para from: {name}")
            if isinstance(q_layer, LSQFakeQuantize):
                a_para += [q_layer.scale]
            if isinstance(q_layer, LSQPlusFakeQuantize):
                a_para += [q_layer.scale, q_layer.zero_point]
        # collect activation quantization params
        if isinstance(q_layer, QuantizeBase) and "post_act_fake_quantize_med" in name:
            if isinstance(q_layer, LSQFakeQuantize):
                a_para += [q_layer.scale]
            if isinstance(q_layer, LSQPlusFakeQuantize):
                a_para += [q_layer.scale, q_layer.zero_point]

    # Set optimizers for quantization parameters of weights and activations
    if len(a_para) != 0:
        a_opt = torch.optim.Adam(a_para, lr=qconfig.recon.scale_lr)
        a_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            a_opt, T_max=qconfig.recon.iters, eta_min=0.0
        )
    else:
        a_opt, a_scheduler = None, None
    w_opt = torch.optim.Adam(w_para)

    # use gd_loss if block // if layer, use mse_loss
    use_gd_loss = isinstance(q_module, QuantizedBlock)

    # Define loss function
    loss_func = LossFunction(
        module=q_module,
        weight=qconfig.recon.weight,
        iters=qconfig.recon.iters,
        b_range=qconfig.recon.b_range,
        warm_up=qconfig.recon.warm_up,
        use_gd_loss=use_gd_loss,
        qconfig=qconfig,
    )

    for i in tqdm(
        range(qconfig.recon.iters),
        desc=f"Reconstruction with GD Loss: {use_gd_loss}...",
    ):
        # Get random index for batch
        batch_idx = torch.randint(
            0, q_block_inputs.size(0), (qconfig.recon.batch_size,)
        )

        fp_block_input = fp_block_inputs[batch_idx].to(device)
        fp_block_output = fp_module(fp_block_input)

        q_block_input = q_block_inputs[batch_idx].to(device)
        q_block_output = q_module(q_block_input)

        # init extra features for block reconstruction's gd_loss
        f_l, f_m, f_h, f_lmh = None, None, None, None
        if isinstance(q_module, QuantizedBlock):
            f_l = q_module.f_l
            f_m = q_module.f_m
            f_h = q_module.f_h
            f_lmh = q_module.f_lmh
            # f_mixed = q_block_output
        # Compute loss
        loss = loss_func(fp_block_output, q_block_output, f_l, f_m, f_h, f_lmh)

        # clear old gradients
        if a_opt:
            a_opt.zero_grad()
        w_opt.zero_grad()

        # back-propagation
        loss.backward()

        # update parameters
        w_opt.step()
        if a_opt:
            a_opt.step()
            a_scheduler.step()

        wandb.log(
            {
                "Block Reconstruction Loss (GD Loss)": loss_func.recon_loss,
                "Block Weight Rounding Loss (AdaRound)": loss_func.round_loss,
                "Total Loss (round+recon)": loss.item(),
            }
        )

    torch.cuda.empty_cache()
    # UPDATE WITH OPTIMIZED PARAMS
    for name, layer in q_module.named_modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            weight_quantizer = layer.weight_fake_quant
            layer.weight.data = weight_quantizer.get_hard_value(layer.weight.data)
            weight_quantizer.adaround = False
        if isinstance(layer, QuantizeBase) and "post_act_fake_quantize" in name:
            layer.drop_prob = 1.0