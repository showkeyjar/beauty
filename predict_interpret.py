# %% coding=utf-8
import sys
import dlib
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from interpret import show
from interpret.perf import RegressionPerf
from interpret.blackbox import LimeTabular
from interpret.blackbox import ShapKernel

"""
实际预测部分
"""
#todo
# 1.lime问题太多，将lime更换成interpret

#%%
predictor_path = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
model = joblib.load('model/beauty.pkl')
df_input = pd.read_csv('data/face/df_input.csv', dtype=np.float64)
df_label = df_input['label'].values

df_input = df_input.drop(['Unnamed: 0', 'Image', 'label'], axis=1)
feature_names = df_input.columns
df_input = df_input.values
print(feature_names)


#%%
def prepare_input(img_path):
    img = dlib.load_rgb_image(img_path)
    dets = detector(img, 1)
    df_image = None
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        f_width = abs(d.right() - d.left())
        f_height = abs(d.bottom() - d.top())
        # print('width:' + str(f_width) + ', height:' + str(f_height))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        face_shape = {}
        for i in range(0, 67):
            for j in range(i + 1, 68):
                face_shape[str(i) + '_' + str(j) + '_x'] = abs(shape.part(i).x - shape.part(j).x) / f_width
                face_shape[str(i) + '_' + str(j) + '_y'] = abs(shape.part(i).y - shape.part(j).y) / f_height
                # print(str(i) + '_' + str(j))
        # shape_size.append(face_shape)
        df_image = pd.DataFrame.from_dict([face_shape])
        break
    return df_image


#%%
def predict(f):
    global detector, predictor, model
    #shape_size = []
    img = dlib.load_rgb_image(f)
    dets = detector(img, 1)
    for k, d in enumerate(dets):
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
        pred = model.predict(df_image)
        break
    return pred


if __name__ == "__main__":
    try:
        test = sys.argv[1]
        mode = sys.argv[2]
    except:
        test = "data/t1.jpg"
        mode = 'shap'
    score = predict(test)
    # result = model.predict(df_input)
    print('beauty score:' + str(score))
    X_test = prepare_input(test)
    y_test = model.predict(X_test)

    if mode == 'blackbox':
        blackbox_perf = RegressionPerf(model.predict).explain_perf(df_input, df_label, name='Blackbox')
        show(blackbox_perf)
    elif mode == 'lime':
        #%% Blackbox explainers need a predict function, and optionally a dataset
        lime = LimeTabular(predict_fn=model.predict, data=df_input, random_state=1)
        #%%Pick the instances to explain, optionally pass in labels if you have them
        lime_local = lime.explain_local(X_test, y_test, name='LIME')
        show(lime_local)
    else:
        #%%
        background_val = np.median(df_input, axis=0).reshape(1, -1)
        #%%
        shap = ShapKernel(predict_fn=model.predict, data=background_val, feature_names=feature_names)
        #%%
        shap_local = shap.explain_local(X_test, y_test, name='SHAP')
        show(shap_local)
