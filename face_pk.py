import os
import dlib
import torch
import numpy as np
from scipy.spatial import distance
from dl.siamese.networks import EmbeddingNet, SiameseNet

"""
todo 颜值pk
基本原理：
通过与目标颜值的差异比较
得到pk结果
"""
cuda = torch.cuda.is_available()


def load_face(f):
    # 加载人脸图像
    img = dlib.load_rgb_image(f)
    return img


def load_model():
    embedding_net = EmbeddingNet()
    model = SiameseNet(embedding_net)
    if cuda:
        model.cuda()
    return model


def extract_embeddings(dataloader, model):
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), 2))
        labels = np.zeros(len(dataloader.dataset))
        k = 0
        for images, target in dataloader:
            if cuda:
                images = images.cuda()
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)
    return embeddings, labels


def face_pk(f1, f2):
    """
    人脸pk
    """
    results = ""
    img1 = load_face(f1)
    img2 = load_face(f2)
    model = load_model()
    face_embeddings1 = extract_embeddings(img1, model)
    face_embeddings2 = extract_embeddings(img2, model)
    # todo compare distances
    results = distance.euclidean(face_embeddings1, face_embeddings2)
    return results


if __name__=="__main__":
    face1 = "data/1.jpg"
    face2 = "data/2.jpg"
    results = face_pk(face1, face2)
    print(results)
