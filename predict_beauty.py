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
df_input = pd.read_csv('/data/face/df_input.csv')
df_input.drop(['Unnamed: 0', 'Image'], axis=1, inplace=True)
print(df_input.columns)


def predict(f):
    """

    :param f:
    :return:
    """
    global detector, predictor, df_input, model
    #shape_size = []
    img = dlib.load_rgb_image(f)
    dets = detector(img, 1)
    for _, d in enumerate(dets):
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        f_width = abs(d.right() - d.left())
        f_height = abs(d.bottom() - d.top())
        #print('width:' + str(f_width) + ', height:' + str(f_height))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        #print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        face_shape = {}
        for i in range(0, 67):
            for j in range(i+1, 68):
                face_shape[str(i) + '_' + str(j) + '_x'] = abs(shape.part(i).x - shape.part(j).x)/f_width
                face_shape[str(i) + '_' + str(j) + '_y'] = abs(shape.part(i).y - shape.part(j).y)/f_height
                #print(str(i) + '_' + str(j))
        #shape_size.append(face_shape)
        df_image = pd.DataFrame.from_dict([face_shape])
        #print(df_image.columns)
        explainer = lime.lime_tabular.LimeTabularExplainer( df_input.values[:,:], mode='regression', training_labels='label', feature_names=df_image.columns, verbose=True)
        #explainer = lime.lime_tabular.LimeTabularExplainer(df_input, mode='regression', training_labels='label')
        exp = explainer.explain_instance(df_image, model.predict)
        pred = exp.score
        #pred = model.predict(df_input)
        break
    return pred


if __name__ == "__main__":
    try:
        test = sys.argv[1]
    except:
        test = "/data/face/Data_Collection/SCUT-FBP-9.jpg"
    score = predict(test)
    print('beauty score:' + str(score))
