import numpy as np
import gc
import os
import logging
import time
import yaml
from tqdm import tqdm
from easydict import EasyDict
import random
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm

logger = logging.getLogger("ptmq")


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        curr_config = config
        curr_path = config_file
        while "root" in curr_config:
            root_path = os.path.dirname(curr_path)
            curr_path = os.path.join(root_path, curr_config["root"])
            with open(curr_path) as r:
                root_config = yaml.load(r, Loader=yaml.FullLoader)
                for k, v in root_config.items():
                    if k not in config:
                        config[k] = v
                curr_config = root_config
    config = EasyDict(config)
    return config


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# hook function
class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """

    pass


class DataSaverHook:
    """
    Forward hook that stores the input and output of a layer/block
    """

    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input, output):
        if self.store_input:
            self.input_store = input
        if self.store_output:
            self.output_store = output
        if self.stop_forward:
            raise StopForwardException()


# load data
def load_data(
    config,
    path="",
    input_size=224,
    batch_size=64,
    num_workers=4,
    pin_memory=True,
    test_resize=256,
):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # train_dir = "../imagenet-mini/train"
    train_dir = os.path.join(path, 'train')
    val_dir = os.path.join(path, "val")

    train_transforms = transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transforms = transforms.Compose(
        [
            transforms.Resize(test_resize),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    """ timm.vit transforms
    Compose(
    Resize(size=248, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    MaybeToTensor()
    Normalize(mean=tensor([0.5000, 0.5000, 0.5000]), std=tensor([0.5000, 0.5000, 0.5000]))
    )
    
    val_transforms = transforms.Compose([
        transforms.Resize(size=248, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=True),
        transforms.CenterCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5000, 0.5000, 0.5000], std=[0.5000, 0.5000, 0.5000])
    ])
    """

    """ timm.regnetx_006 transforms
    Compose(
        Resize(size=256, interpolation=bicubic, max_size=None, antialias=True)
        CenterCrop(size=(224, 224))
        MaybeToTensor()
        Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
    )
    """

    if config.model.type.startswith(
        ("vit", "deit", "regnet")
    ):
        model = timm.create_model("regnetx_006", pretrained=True)
        data_config = timm.data.resolve_model_data_config(model)
        del model
        gc.collect()
        val_transforms = timm.data.create_transform(**data_config, is_training=False)
        train_transforms = timm.data.create_transform(**data_config, is_training=True)
    
    print(val_transforms)
    
    train_dataset = datasets.ImageFolder(train_dir, train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, val_transforms)

    # only input 4% of the data - for faster debugging
    # val_dataset = torch.utils.data.Subset(
    #     val_dataset, np.arange(0, len(val_dataset), 25)
    # )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info("Finished loading dataset")
    return train_loader, val_loader


class AverageMeter(object):
    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)

    batch_time = AverageMeter("Time", ":6.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")
    progress = ProgressMeter(len(val_loader), [batch_time, top1, top5], prefix="Test: ")

    model.eval()
    t = time.time()
    for i, (images, target) in tqdm(
        enumerate(val_loader), desc="imagenet_val", total=len(val_loader)
    ):
        images = images.to(device)
        target = target.to(device)

        output = model(images)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        batch_time.update(time.time() - t)
        t = time.time()

        if i % print_freq == 0:
            progress.display(i)

    logger.info(
        " * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}".format(top1=top1, top5=top5)
    )
    return top1.avg, top5.avg
