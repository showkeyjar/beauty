# %% coding=utf-8
import sys
import dlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib

"""
实际预测部分
"""
#todo
# 1.lime问题太多，将lime更换成interpret


predictor_path = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
model = joblib.load('model/beauty.pkl')


def prepare_input(img, face):
    f_width = abs(face.right() - face.left())
    f_height = abs(face.bottom() - face.top())
    shape = predictor(img, face)
    # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
    face_shape = {}
    for i in range(0, 67):
        for j in range(i + 1, 68):
            face_shape[str(i) + '_' + str(j) + '_x'] = abs(shape.part(i).x - shape.part(j).x) / f_width
            face_shape[str(i) + '_' + str(j) + '_y'] = abs(shape.part(i).y - shape.part(j).y) / f_height
            # print(str(i) + '_' + str(j))
    # shape_size.append(face_shape)
    df_image = pd.DataFrame.from_dict([face_shape])
    return df_image


def predict(f):
    global detector, predictor, model
    #shape_size = []
    img = dlib.load_rgb_image(f)
    dets = detector(img, 1)
    # 仅预测第一张人脸
    d = dets[0]
    df_image = prepare_input(img, d)
    #print(df_image.columns)
    pred = model.predict(df_image)
    # 这里由于使用的是回归模型，所以对分数区间做限制
    pred = 0 if pred<0 else pred
    pred = 5 if pred>5 else pred
    return pred


if __name__ == "__main__":
    try:
        test = sys.argv[1]
    except:
        # 修改这里的图片路径
        test = "data/2.jpg"
    score = predict(test)
    print('beauty score:' + str(score))
