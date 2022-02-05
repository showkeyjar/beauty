import glob
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from solo.args.setup import parse_args_linear
from solo.methods.base import BaseMethod
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
    args = parse_args_linear()

    assert args.backbone in BaseMethod._SUPPORTED_BACKBONES
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
    }[args.backbone]

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


def predict(model, x):
    # model.backbone.eval()
    # output = model(x)["logits"]
    model.eval()
    with torch.no_grad():
        output = model(x)
    # predicted = int(torch.max(output.data, 1)[1].numpy())
    return output


def load_one(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name).convert("RGB")
    transform_train=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])
    image = transform_train(image).float()
    image = image.unsqueeze(0)
    return image


def save(model):
    torch.save(model, "byol.pt")


if __name__=="__main__":
    m = load_model()
    save(m)
    test_files = glob.glob("/mnt/data/soft/skin/cheek/*.jpg")
    all_img = []
    for f in test_files:
        img=load_one(f)
        output = predict(m, img)[0].numpy()
        print(output, ":", f)
        se = pd.Series({'file': f, 'data': output})
        all_img.append(se)
    # img_t = torch.stack(all_img)
    # outputs = predict(m, img_t)
    # print(outputs)
    df_all = pd.DataFrame(all_img)
    df_all.to_feather("/mnt/data/soft/skin/cheek_boyl.ftr")
