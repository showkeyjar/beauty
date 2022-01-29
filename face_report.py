# %% coding=utf-8
import sys
import shap
import warnings
from sklearn.externals import joblib
from gen_report import gen_report
from shap.common import convert_to_link, Instance, Model, Data, DenseData, Link
from feature.tools import *
"""
预测解释
"""

# %%
model = joblib.load('model/beauty_1.pkl')
with open('model/explainer_1.pkl', 'rb') as f:
    explainer = dill.load(f)
df_explain = pd.read_csv('model/explain.csv')
df_explain['key'] = df_explain['key'].astype(str)
with open('model/best_values.pkl', 'rb') as f:
    se_best = dill.load(f)


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
    result_df = pd.DataFrame({'feature': exps.data.group_names, 'effect': ensure_not_numpy(exps.effects),
                              'value': exps.instance.group_display_values})
    result_df = result_df[result_df['effect'] != 0].reset_index()
    return result_df


def get_feature_name(x):
    points = x.split('_')
    try:
        feature_name = points[0] + '_' + points[1]
    except:
        feature_name = x
    return feature_name


def get_explain(x, gen_pic=True):
    """
    获取解释
    :param x:
    :param gen_pic: 是否生成对应碎片
    :return:
    """
    global df_explain
    points = x.split('_')
    if 'skin' in points:
        exp = '皮肤'
    else:
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
    # 生成对应碎片图像
    feature_name = get_feature_name(x)
    if gen_pic:
        gen_feature_pic(feature_name)
    return exp


def predict(im_path):
    X_test = prepare_input(im_path)
    Y_test = 0
    if X_test is not None:
        Y_test = model.predict(X_test)
    return Y_test


def gen_default_report(X_test):
    """
    生成默认报告
    (将优劣部位生成图像碎片)
    :param X_test:
    :return:
    """
    global se_best
    se_test = X_test.iloc[0]
    result = (se_best - se_test).abs()
    result.sort_values(ascending=False, inplace=True)
    good_points = list(result[:5].index)
    bad_points = list(result[-5:].index)
    good_effect = list(map(get_explain, good_points))
    bad_effect = list(map(get_explain, bad_points))
    good_feature = list(map(get_feature_name, good_points))
    bad_feature = list(map(get_feature_name, bad_points))
    good_str = ",".join(good_effect)
    bad_str = ",".join(bad_effect)
    return good_str, bad_str, good_feature, bad_feature


def gen_shap_report(X_test):
    """
    生成shap报告
    shap 内存消耗过大
    :param X_test:
    :return:
    """
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
        good_str = str(good_effect['explain'].values[0:10]).replace("'", "")
    if bad_effect is not None:
        bad_str = str(bad_effect['explain'].values[0:10]).replace("'", "")
    return good_str, bad_str


def gen_report_file(im_path, name='t1', type='default'):
    """
    生成报告
    :param im_path:
    :param name:
    :param type: 报告类型
    :return:
    """
    X_test = prepare_input(im_path)
    Y_test = 0
    if X_test is not None:
        Y_test = model.predict(X_test)
    params = []
    web_path = "/" + im_path
    params.append(web_path)
    print('beauty score:' + str(Y_test))
    params.append(Y_test)
    if type == 'default':
        good_effect, bad_effect, good_features, bad_features = gen_default_report(X_test)
    else:
        good_effect, bad_effect = gen_shap_report(X_test)
    try:
        feature_path = web_path.split('.')[0] + "/"
    except:
        feature_path = "/static/uploads/tmp/"
    good_features = [feature_path + x + ".jpg" for x in good_features]
    bad_features = [feature_path + x + ".jpg" for x in bad_features]
    if good_effect is not None:
        params.append(good_effect)
        params.extend(good_features)
        print('您的优势部位:' + good_effect)
    else:
        params.append(None)
        params.extend([None]*5)
    if bad_effect is not None:
        params.append(bad_effect)
        params.extend(bad_features)
        print('您的欠缺部位:' + bad_effect)
    else:
        params.append(None)
        params.extend([None]*5)
    gen_report(name, params)


if __name__ == "__main__":
    try:
        test = sys.argv[1]
        mode = sys.argv[2]
    except:
        test = "img/t1.jpg"
        mode = 'shap'
    X_test = prepare_input(test)
    y_test = model.predict(X_test)
    print('beauty score:' + str(y_test))
    shap_values = explainer.shap_values(X_test)
    print('gen explain')
    result = force_df(explainer.expected_value, shap_values[0, :], X_test)
    result['explain'] = result['feature'].apply(get_explain)
    good_effect = result[result['effect'] > 0.01].sort_values('effect', ascending=False).reset_index()
    bad_effect = result[result['effect'] < 0.01].sort_values('effect').reset_index()
    try:
        good = str(good_effect['explain'][0:10].values)
        bad = str(bad_effect['explain'][0:10].values)
    except Exception as e:
        print(e)
        good = ''
        bad = ''
    print('您的优势部位:' + good)
    print('您的欠缺部位:' + bad)
