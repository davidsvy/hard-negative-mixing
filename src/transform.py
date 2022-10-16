import torch
import torch.nn as nn
import torchvision.transforms as T


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


dict_interp = {
    'bicubic': T.InterpolationMode.BICUBIC,
    'bilinear': T.InterpolationMode.BILINEAR,
    'box': T.InterpolationMode.BOX,
    'hamming': T.InterpolationMode.HAMMING,
    'lanczos': T.InterpolationMode.LANCZOS,
    'nearest': T.InterpolationMode.NEAREST,
}


def transform_inner_train(crop_size=112, min_area=0.5, interpolation='bicubic'):
    return T.RandomResizedCrop(
        size=crop_size,
        scale=(min_area, 1),
        interpolation=dict_interp[interpolation],
    )


def transform_inner_val(crop_size=112, interpolation='bicubic'):
    inception_size = int((256 / 224) * crop_size)

    transform = T.Compose([
        T.Resize(
            size=inception_size,
            interpolation=dict_interp[interpolation],
        ),
        T.CenterCrop(crop_size),
        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        ),
    ])

    return transform


def float32_to_uint8(x):
    return (torch.clamp(x, 0, 1) * 255).to(torch.uint8)


def uint8_to_float32(x):
    return torch.clamp(x / 255.0, 0, 1).to(torch.float32)


def transform_outer_train(randaugment_m=9, randaugment_n=2, erase_prob=0.25):
    transform = [T.RandomHorizontalFlip()]

    if randaugment_m > 0 and randaugment_n > 0:
        transform += [
            T.Lambda(float32_to_uint8),
            # T.ConvertImageDtype(torch.uint8),
            T.RandAugment(
                magnitude=randaugment_m,
                num_ops=randaugment_n,
                interpolation=T.InterpolationMode.BILINEAR,
                fill=0,
            ),
            T.Lambda(uint8_to_float32),
            # T.ConvertImageDtype(torch.float32),
            T.RandomApply([T.GaussianBlur(3)], p=0.5),
        ]

    if erase_prob > 0:
        transform.append(
            T.RandomErasing(
                p=erase_prob,
                scale=(0.02, 0.2),
                value=0,  # 'random',
            )
        )

    transform += [
        T.Normalize(
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )
    ]

    transform = T.Compose(transform)

    return transform


class Transform_Contrastive(nn.Module):

    def __init__(self, transform_q=None, transform_k=None):
        super(Transform_Contrastive, self).__init__()
        self.transform_q = transform_q
        self.transform_k = transform_k

    def forward(self, image, **kwargs):
        # image -> [B, 2, C, H, W]

        q = self.transform_q(image[:, 0])
        k = self.transform_k(image[:, 1])

        return q, k


def build_transform_contrastive(args):
    transform = transform_outer_train()
    transform_contrastive = Transform_Contrastive(
        transform_q=transform, transform_k=transform
    )

    return transform_contrastive
