import torch
import random
import argparse
from torchvision import models, datasets, transforms
import torchvision.transforms as transforms

"""
BarlowTwins Predictor
"""


def load_model(model_type="face"):
    # gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu = 0
    model = models.resnet50().cuda(gpu)
    state_dict = torch.load("model/barlowtwins/" + model_type + ".pth", map_location='cpu')
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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


def predict_img(val_dataset, args):
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)
    model = load_model(args.model_type)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    out_scores = []
    for images, target in val_loader:
        output = model(images.cuda(0, non_blocking=True))
        acc1, acc5 = accuracy(output, target.cuda(0, non_blocking=True), topk=(1, 5))
        out_scores.append([acc1, acc5])
    # output2 = predict(t2, model, device)
    return out_scores


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Barlow Twins Predict')
    parser.add_argument("-d", default="data/test/")
    args = parser.parse_args()
    args.ngpus_per_node = torch.cuda.device_count()
    # single-node distributed training
    args.rank = 0
    args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
    args.world_size = args.ngpus_per_node
    args.batch_size = 256
    args.workers = 8
    args.model_type = "score"

    img_dir = str(args.d)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_dataset = datasets.ImageFolder(img_dir, transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    result = predict_img(val_dataset, args)
    print(result)
