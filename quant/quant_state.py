import logging
from quant.fake_quant import QuantizeBase

logger = logging.getLogger("quantization")


def enable_calib_without_quant(model, quantizer_type="fake_quant"):
    logger.info(f"Enable observer and disable quantize for {quantizer_type}")
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if quantizer_type not in name:
                logger.debug(f"The except_quantizer is {name}")
                module.disable_observer()
                module.disable_fake_quant()
            else:
                logger.debug(f"Enable observer and disable quant: {name}")
                module.enable_observer()
                module.disable_fake_quant()


def enable_quantization(model, quantizer_type="fake_quant"):
    logger.info("Disable observer and enable quant")

    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            if quantizer_type not in name:
                print(f"The except_quantizer is {name}")
                module.disable_observer()
                module.disable_fake_quant()
            else:
                print(f"Disable observer and enable quant: {name}")
                module.disable_observer()
                module.enable_fake_quant()


def disable_all(model):
    logger.info("Disable observer and disable quant")
    for name, module in model.named_modules():
        if isinstance(module, QuantizeBase):
            logger.debug(f"Disable observer and disable quant: {name}")
            module.disable_observer()
            module.disable_fake_quant()
