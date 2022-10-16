import timm
import torch.nn as nn


def build_model(arch, n_classes, mlp_head=False):
    if not arch in timm.list_models():
        raise ValueError(f'Unknown architecture: {arch}')

    model = timm.create_model(
        model_name=arch, pretrained=False, num_classes=n_classes)

    if mlp_head:
        in_features = model.get_classifier().in_features
        hidden_features = 512

        model.classifier = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, n_classes),
        )

    return model
