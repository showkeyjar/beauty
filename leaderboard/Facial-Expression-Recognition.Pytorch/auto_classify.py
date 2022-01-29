import os
import shutil
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import transforms as transforms
from skimage import io
from skimage.transform import resize
from nets import *
"""
自动给人脸分类

https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
"""
cut_size = 44
class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
record_file = "../../data/face_record.h5"
try:
    df_record = pd.read_hdf(record_file, "df")
except Exception as e:
    df_record = None
    print(e)


transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

net = VGG('VGG19')

if torch.cuda.is_available():
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'))
    net.load_state_dict(checkpoint['net'])
    net.cuda()
else:
    checkpoint = torch.load(os.path.join('FER2013_VGG19', 'PrivateTest_model.t7'), map_location="cpu")
    net.load_state_dict(checkpoint['net'])
net.eval()


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def predict(img_path):
    global net, class_names
    raw_img = io.imread(img_path)
    gray = rgb2gray(raw_img)
    gray = resize(gray, (48, 48), mode='symmetric').astype(np.uint8)

    img = gray[:, :, np.newaxis]
    img = np.concatenate((img, img, img), axis=2)
    img = Image.fromarray(img)
    inputs = transform_test(img)

    ncrops, c, h, w = np.shape(inputs)

    inputs = inputs.view(-1, c, h, w)
    if torch.cuda.is_available():
        inputs = inputs.cuda()
    inputs = Variable(inputs, volatile=True)
    outputs = net(inputs)

    outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops
    score = F.softmax(outputs_avg)
    print("output score:", score)
    _, predicted = torch.max(outputs_avg.data, 0)

    if torch.cuda.is_available():
        pred_data = predicted.cpu().numpy()
    else:
        pred_data = predicted.numpy()
    print("pred result:", str(class_names[int(pred_data)]))
    return str(class_names[int(pred_data)])


def classfiy_img(se):
    img_path = "../../" + se['img_path'].replace("\\", "/")
    img_name = img_path.split("/")[-1]
    class_name = predict(img_path)
    new_path = "../../data/class/" + class_name + "/"
    try:
        os.makedirs(new_path)
    except Exception:
        pass
    try:
        shutil.copyfile(img_path, new_path + img_name)
    except Exception as e:
        print(e)
    print("process file:", new_path + img_name)
    return pd.Series({'class': class_name, 'img_path': new_path.replace("../../", "") + img_name})


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=int)
    args = parser.parse_args()
    force = False
    if args.f:
        force = True
    if force:
        df_record.loc[:, ['class', 'img_path']] = df_record.apply(classfiy_img, axis=1)
    else:
        df_record.loc[df_record['class'].isna(), ['class', 'img_path']] = df_record[df_record['class'].isna()].apply(classfiy_img, axis=1)
    df_record.to_hdf(record_file, "df")
