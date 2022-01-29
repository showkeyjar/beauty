import torch
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from solo.utils.backbones import (
    swin_base,
    swin_large,
    swin_small,
    swin_tiny,
    vit_base,
    vit_large,
    vit_small,
    vit_tiny,
)


def load_model():
    args = {}
    args["dataset"] = "custom"
    args["backbone"] = "resnet18"
    args["data_dir"] = "./"
    args["train_dir"] = "./"
    args["gpus"] = 0
    args["sync_batchnorm"] = True
    args["precision"] = 16
    args["optimizer"] = "sgd"
    args["lars"] = True
    args["lr"] = 0.1
    args["weight_decay"] = 1e-5
    args["batch_size"] = 128
    args["name"] = "general-linear-eval"
    args["pretrained_feature_extractor"] = "trained_models/byol/offline-w7ez810q/byol-400ep-custom-offline-w7ez810q-ep=399.ckpt"
    args["project"] = "self-supervised"

    # assert args.backbone in BaseMethod._SUPPORTED_BACKBONES
    backbone_model = {
        "resnet18": resnet18,
        "resnet50": resnet50,
        "vit_tiny": vit_tiny,
        "vit_small": vit_small,
        "vit_base": vit_base,
        "vit_large": vit_large,
        "swin_tiny": swin_tiny,
        "swin_small": swin_small,
        "swin_base": swin_base,
        "swin_large": swin_large,
    }[args["backbone"]]

    # initialize backbone
    kwargs = args.backbone_args
    cifar = kwargs.pop("cifar", False)
    # swin specific
    if "swin" in args.backbone and cifar:
        kwargs["window_size"] = 4

    backbone = backbone_model(**kwargs)
    if "resnet" in args.backbone:
        # remove fc layer
        backbone.fc = nn.Identity()
        if cifar:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
            backbone.maxpool = nn.Identity()

    assert (
        args.pretrained_feature_extractor.endswith(".ckpt")
        or args.pretrained_feature_extractor.endswith(".pth")
        or args.pretrained_feature_extractor.endswith(".pt")
    )
    ckpt_path = args.pretrained_feature_extractor

    state = torch.load(ckpt_path)["state_dict"]
    for k in list(state.keys()):
        if "encoder" in k:
            raise Exception(
                "You are using an older checkpoint."
                "Either use a new one, or convert it by replacing"
                "all 'encoder' occurances in state_dict with 'backbone'"
            )
        if "backbone" in k:
            state[k.replace("backbone.", "")] = state[k]
        del state[k]
    backbone.load_state_dict(state, strict=False)
    # del args.backbone
    # line_model = LinearModel(backbone, **args.__dict__)
    # line_model.num_classes = 1024
    return backbone


def save(model):
    torch.save(model, "byol.pt")


if __name__=="__main__":
    m = load_model()
    save(m)
