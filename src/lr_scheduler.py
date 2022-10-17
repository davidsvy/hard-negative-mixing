
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_scheduler(args, optimizer):
    steps_total = args.steps
    steps_warmup = int(0.1 * args.steps)

    lr_scheduler = CosineLRScheduler(
        optimizer,
        t_initial=steps_total,
        cycle_mul=1.,
        lr_min=args.lr_min,
        warmup_lr_init=args.lr_warmup,
        warmup_t=steps_warmup,
        cycle_limit=1,
        t_in_epochs=False,
    )

    return lr_scheduler
