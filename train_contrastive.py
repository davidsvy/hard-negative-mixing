import datetime
import json
import math
import os
import time

import torch.nn as nn
import torch.nn.functional as F

from src.args import parse_args
from src.dataset import build_loader
from src.logger import create_logger
from src.lr_scheduler import build_scheduler
from src.moco import build_moco
from src.optimizer import build_optimizer
from src.transform import build_transform_contrastive

from src.utils import (
    accuracy,
    Average_Meter,
    count_parameters,
    get_device,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.resume:
        assert os.path.isfile(args.resume)

    os.makedirs(args.dir_out, exist_ok=True)
    logger = create_logger(args.dir_out)

    device = get_device(args=args, logger=logger)

    path_args = os.path.join(args.dir_out, 'args.yaml')
    with open(path_args, 'w') as file_args:
        json.dump(args.__dict__, file_args, indent=2)

    logger.info(f'Full args saved to {path_args}')
    logger.info(args.__dict__)

    dataloader = build_loader(args)
    transform = build_transform_contrastive(args)

    model = build_moco(args).to(device)
    loss_fn = F.cross_entropy

    n_parameters = count_parameters(model)
    logger.info(f'Number of params: {n_parameters}')

    optimizer = build_optimizer(args=args, model=model)
    lr_scheduler = build_scheduler(args=args, optimizer=optimizer)

    global_step = 0

    if args.resume:
        global_step = load_checkpoint(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
        )

    n_steps_per_epoch = len(dataloader)
    n_epochs = math.ceil(
        (args.steps - global_step) / n_steps_per_epoch)
    logger.info(
        f'Training for {n_epochs} epoch(s) with {n_steps_per_epoch} steps per epoch.')

    start_time = time.time()

    for epoch in range(1, n_epochs + 1):
        logger.info('#' * 60)
        logger.info(
            f'Step {global_step}: starting epoch [{epoch}/{n_epochs}]...')

        global_step = train_one_epoch(
            args=args,
            model=model,
            loss_fn=loss_fn,
            dataloader=dataloader,
            transform=transform,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            logger=logger,
            device=device,
            global_step=global_step,
            global_start_time=start_time,
        )

        if global_step < 0:
            break

    total_time = time.time() - start_time
    total_time = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('#' * 60)
    logger.info(f'Total training time: {total_time}')


def train_one_epoch(
    args, model, loss_fn, dataloader, transform, optimizer, lr_scheduler,
    logger, device, global_step, global_start_time,
):
    model.train()
    last_iter = False

    acc1_meter, acc5_meter = Average_Meter(), Average_Meter()
    loss_meter = Average_Meter()
    start_time = time.time()

    for local_step, (batch, file_idxs) in enumerate(dataloader, 1):
        for param in model.parameters():
            param.grad = None

        batch = batch.to(device)
        file_idxs = file_idxs.to(device)

        q, k = transform(batch)
        outputs, labels = model(
            q=q, k=k, file_idxs=file_idxs, step=global_step)
        loss = loss_fn(outputs, labels)

        loss.backward()

        if args.clip_grad:
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        optimizer.step()
        lr_scheduler.step_update(global_step)

        batch_size = labels.shape[0]
        loss_meter.update_avg(val=loss.item(), n=batch_size)

        acc1, acc5 = accuracy(
            output=outputs.detach().float(), target=labels, topk=(1, 5))
        acc1_meter.update_avg(val=acc1.item(), n=batch_size)
        acc5_meter.update_avg(val=acc5.item(), n=batch_size)

        if global_step >= args.steps - 1:
            last_iter = True

        if (global_step + 1) % args.steps_save == 0 or last_iter:
            save_checkpoint(
                args=args,
                step=global_step,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                logger=logger,
            )

        if (global_step + 1) % args.steps_print == 0:
            lr = optimizer.param_groups[0]['lr']

            so_far = time.time() - global_start_time
            avg_time = (time.time() - start_time) / local_step
            eta = avg_time * (args.steps - global_step)

            so_far = datetime.timedelta(seconds=int(so_far))
            eta = datetime.timedelta(seconds=int(eta))

            logger.info('#' * 60)
            logger.info(f'Step [{global_step + 1}/{args.steps}]')
            logger.info(f'Time -> So-far: {so_far}, ETA: {eta}')
            logger.info(
                f'Metrics -> ACC1: {acc1_meter.avg:.2f}, ACC5: {acc5_meter.avg:.2f}')
            logger.info(f'Other -> Loss: {loss_meter.avg:.5f}, LR: {lr:.6f}')

            acc1_meter.reset()
            acc5_meter.reset()
            loss_meter.reset()

        if last_iter:
            return -1

        global_step += 1

    return global_step


if __name__ == '__main__':
    main()
