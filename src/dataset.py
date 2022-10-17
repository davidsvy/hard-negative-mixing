import os
import torch
import torchvision

from src.transform import transform_inner_train


class Dataset_Contrastive(torchvision.datasets.Caltech256):

    def __getitem__(self, idx):
        path = os.path.join(
            self.root,
            '256_ObjectCategories',
            self.categories[self.y[idx]],
            f'{self.y[idx] + 1:03d}_{self.index[idx]:04d}.jpg',
        )

        image = torchvision.io.read_image(
            path, mode=torchvision.io.ImageReadMode.RGB) / 255.0

        image1 = self.transform(image)
        image2 = self.transform(image)
        # image1, image2 -> [C, H, W]
        image = torch.stack((image1, image2), dim=0)
        # image -> [2, C, H, W]

        return image, idx


def build_loader(args):

    transform = transform_inner_train(crop_size=args.img_size)

    dataset = Dataset_Contrastive(
        root=args.dir_data,
        transform=transform,
        download=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    return dataloader
