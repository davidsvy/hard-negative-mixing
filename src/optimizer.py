import torch


def build_optimizer(args, model):
    """
    Build optimizer, set weight decay of normalization to 0 by default.
    """
    skip = {}
    skip_keywords = []

    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = set_weight_decay(
        model=model,
        skip_list=skip,
        skip_keywords=skip_keywords,
    )
    
    eps = 1e-8
    betas = (0.9, 0.999)
    weight_decay = 0.01

    optimizer = torch.optim.AdamW(
        params=parameters,
        eps=eps,
        betas=betas,
        lr=args.lr_base,
        weight_decay=weight_decay,
    )
    
    return optimizer



def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if (
            len(param.shape) == 1
            or name.endswith('.bias')
            or (name in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)

        else:
            has_decay.append(param)

    parameters = [
        {'params': has_decay},
        {'params': no_decay, 'weight_decay': 0.}
    ]

    return parameters


def check_keywords_in_name(name, keywords=()):
    isin = False

    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin
