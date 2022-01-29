# %% coding=utf-8
import sys
import shap
import dlib
import dill
import warnings
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from gen_report import gen_report
from shap.common import convert_to_link, Instance, Model, Data, DenseData, Link

"""
预测解释 shap
"""

#%%
predictor_path = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
model = joblib.load('model/beauty.pkl')
#explainer = joblib.load('model/explainer.pkl')
with open('model/explainer.pkl', 'rb') as f:
    explainer = dill.load(f)
df_input = pd.read_csv('data/face/df_input.csv', dtype=np.float64)
df_label = df_input['label'].values
df_input = df_input.drop(['Unnamed: 0', 'Image', 'label'], axis=1)
feature_names = df_input.columns
df_input = df_input.values
#print(feature_names)
df_explain = pd.read_csv('model/explain.csv')
df_explain['key'] = df_explain['key'].astype(str)


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


class Explanation:
    def __init__(self):
        pass


class AdditiveExplanation(Explanation):
    def __init__(self, base_value, out_value, effects, effects_var, instance, link, model, data):
        self.base_value = base_value
        self.out_value = out_value
        self.effects = effects
        self.effects_var = effects_var
        assert isinstance(instance, Instance)
        self.instance = instance
        assert isinstance(link, Link)
        self.link = link
        assert isinstance(model, Model)
        self.model = model
        assert isinstance(data, Data)
        self.data = data


def ensure_not_numpy(x):
    if isinstance(x, bytes):
        return x.decode()
    elif isinstance(x, np.str):
        return str(x)
    elif isinstance(x, np.generic):
        return float(x.item())
    else:
        return x


def force_df(base_value, shap_values, features=None, feature_names=None, out_names=None, link="identity",
               plot_cmap="RdBu", matplotlib=False, show=True, figsize=(20, 3), ordering_keys=None,
               ordering_keys_time_format=None,
               text_rotation=0):
    # auto unwrap the base_value
    if type(base_value) == np.ndarray and len(base_value) == 1:
        base_value = base_value[0]
    if (type(base_value) == np.ndarray or type(base_value) == list):
        if type(shap_values) != list or len(shap_values) != len(base_value):
            raise Exception("In v0.20 force_plot now requires the base value as the first parameter! " \
                            "Try shap.force_plot(explainer.expected_value, shap_values) or " \
                            "for multi-output models try " \
                            "shap.force_plot(explainer.expected_value[0], shap_values[0]).")
    assert not type(shap_values) == list, "The shap_values arg looks looks multi output, try shap_values[i]."
    link = convert_to_link(link)
    if type(shap_values) != np.ndarray:
        return shap_values
    # convert from a DataFrame or other types
    if str(type(features)) == "<class 'pandas.core.frame.DataFrame'>":
        if feature_names is None:
            feature_names = list(features.columns)
        features = features.values
    elif str(type(features)) == "<class 'pandas.core.series.Series'>":
        if feature_names is None:
            feature_names = list(features.index)
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif features is not None and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None
    if len(shap_values.shape) == 1:
        shap_values = np.reshape(shap_values, (1, len(shap_values)))
    if out_names is None:
        out_names = ["output value"]
    elif type(out_names) == str:
        out_names = [out_names]
    if shap_values.shape[0] == 1:
        if feature_names is None:
            feature_names = [shap.labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]
        if features is None:
            features = ["" for _ in range(len(feature_names))]
        if type(features) == np.ndarray:
            features = features.flatten()
        # check that the shape of the shap_values and features match
        if len(features) != shap_values.shape[1]:
            msg = "Length of features is not equal to the length of shap_values!"
            if len(features) == shap_values.shape[1] - 1:
                msg += " You might be using an old format shap_values array with the base value " \
                       "as the last column. In this case just pass the array without the last column."
            raise Exception(msg)
        instance = Instance(np.zeros((1, len(feature_names))), features)
        exps = AdditiveExplanation(
            base_value,
            np.sum(shap_values[0, :]) + base_value,
            shap_values[0, :],
            None,
            instance,
            link,
            Model(None, out_names),
            DenseData(np.zeros((1, len(feature_names))), list(feature_names))
        )
    else:
        if matplotlib:
            raise Exception("matplotlib = True is not yet supported for force plots with multiple samples!")
        if shap_values.shape[0] > 3000:
            warnings.warn("shap.force_plot is slow for many thousands of rows, try subsampling your data.")
        exps = []
        for i in range(shap_values.shape[0]):
            if feature_names is None:
                feature_names = [shap.labels['FEATURE'] % str(i) for i in range(shap_values.shape[1])]
            if features is None:
                display_features = ["" for i in range(len(feature_names))]
            else:
                display_features = features[i, :]
            instance = Instance(np.ones((1, len(feature_names))), display_features)
            e = AdditiveExplanation(
                base_value,
                np.sum(shap_values[i, :]) + base_value,
                shap_values[i, :],
                None,
                instance,
                link,
                Model(None, out_names),
                DenseData(np.ones((1, len(feature_names))), list(feature_names))
            )
            exps.append(e)
    result_df = pd.DataFrame({'feature': exps.data.group_names, 'effect': ensure_not_numpy(exps.effects), 'value': exps.instance.group_display_values})
    result_df = result_df[result_df['effect'] != 0].reset_index()
    return result_df


def get_explain(x):
    global df_explain
    points = x.split('_')
    exp = ''
    for p in points:
        if p != 'x' and p != 'y':
            try:
                exp += df_explain[df_explain['key'] == p]['explain'].values[0]
            except:
                exp += ''
            exp += '_'
        if p == 'x':
            exp += '宽'
        elif p == 'y':
            exp += '高'
    return exp


def gen_report(im_path):
    X_test = prepare_input(im_path)
    Y_test = model.predict(X_test)
    params = []
    print('beauty score:' + str(Y_test))
    params.append(Y_test)
    shap_values = explainer.shap_values(X_test)
    print('gen explain')
    result = force_df(explainer.expected_value, shap_values[0, :], X_test)
    result['explain'] = result['feature'].apply(get_explain)
    try:
        good_effect = result[result['effect'] > 0.01].sort_values('effect', ascending=False).reset_index()
    except:
        good_effect = None
    try:
        bad_effect = result[result['effect'] < 0.01].sort_values('effect').reset_index()
    except:
        bad_effect = None
    if good_effect is not None:
        good_str = str(good_effect[0:10,'explain'].values)
        params.append(good_str)
        print('您的优势部位:' + good_str)
    if bad_effect is not None:
        bad_str = str(bad_effect[0:10, 'explain'].values)
        params.append(bad_str)
        print('您的欠缺部位:' + bad_str)
    gen_report('t1', params)


if __name__ == "__main__":
    try:
        test = sys.argv[1]
        mode = sys.argv[2]
    except:
        test = "img/t1.jpg"
        mode = 'shap'
    # result = model.predict(df_input)
    X_test = prepare_input(test)
    y_test = model.predict(X_test)
    print('beauty score:' + str(y_test))
    shap_values = explainer.shap_values(X_test)
    print('gen explain')
    result = force_df(explainer.expected_value, shap_values[0, :], X_test)
    result['explain'] = result['feature'].apply(get_explain)
    good_effect = result[result['effect'] > 0.01].sort_values('effect', ascending=False).reset_index()
    bad_effect = result[result['effect'] < 0.01].sort_values('effect').reset_index()
    print('您的优势部位:' + str(good_effect[0:10,'explain'].values))
    print('您的欠缺部位:' + str(bad_effect[0:10,'explain'].values))
