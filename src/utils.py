import numpy as np
import os
import random

import torch
import torch.nn.functional as F


def load_checkpoint(args, model, optimizer, lr_scheduler, logger):
    logger.info(
        f'==============> Resuming from {args.resume}....................')

    checkpoint = torch.load(args.resume, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)

    step = checkpoint['step']

    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        logger.info(
            f"=> loaded successfully '{args.resume}' (step {step})")

    del checkpoint
    torch.cuda.empty_cache()

    return step + 1


def save_checkpoint(args, step, model, optimizer, lr_scheduler, logger):
    save_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'step': step,
        'args': args,
    }

    save_path = os.path.join(args.put, f'ckpt_{step}.pth')
    logger.info('#' * 60)
    logger.info(f'{save_path} saving...')
    torch.save(save_state, save_path)
    logger.info(f'{save_path} saved!')


def set_seed(seed_val):
    if not seed_val is None:
        os.environ['PYTHONHASHSEED'] = str(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


def get_device(args=None, logger=None):
    log_fn = print if logger is None else logger.info

    if not torch.cuda.is_available():
        device = torch.device('cpu')
        log_fn('GPU unavailable.')
        return device

    if args is None or not hasattr(args, 'device') or args.device is None:
        gpu_idx = 0
    else:
        gpu_idx = args.device

    device = torch.device(f'cuda:{gpu_idx}')
    log_fn(f'Available GPU: {torch.cuda.get_device_name(gpu_idx)}.')

    return device


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


class Average_Meter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update_sum(self, val, n=1):
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

    def update_avg(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


@torch.no_grad()
def accuracy(output, target, topk=(1, 5), *args, **kwargs):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    res = torch.cat(res, dim=0)

    return res
