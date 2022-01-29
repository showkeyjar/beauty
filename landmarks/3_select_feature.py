import numpy as np
import pandas as pd
from MLFeatureSelection import sequence_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

"""
MLFeatureSelection筛选特征
https://github.com/duxuhao/Feature-Selection
"""

sf = sequence_selection.Select(Sequence=True, Random=True, Cross=False)

df = pd.read_csv('data/face/df_input.csv')

try:
    df.drop(['Unnamed: 0', 'file'], axis=1, inplace=True)
    df['Rating'] = df['Rating'].round().astype(int)
    df = df.replace([np.nan, np.inf, -np.inf], 0)
    df.dropna(axis=0, inplace=True)
    df.fillna(0, inplace=True)
except Exception as e:
    print(e)
    df = None

sf.ImportDF(df, label='Rating')  # import dataframe and label


def lossfunction(y_pred, y_test):
    """define your own loss function with y_pred and y_test
    return score
    """
    return np.mean(y_pred == y_test)


def validation(X, y, features, clf, lossfunction):
    totaltest = []
    kf = KFold(5)
    for train_index, test_index in kf.split(X):
        # print(train_index)
        # print(test_index)
        X_train, X_test = X.reindex(train_index)[features], X.reindex(test_index)[features]
        y_train, y_test = y.reindex(train_index), y.reindex(test_index)
        # clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=50)
        # todo 检查 ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
        clf.fit(X_train, y_train)
        totaltest.append(lossfunction(y_test, clf.predict(X_test)))
    return np.mean(totaltest), clf


#notusable = ['file']
notusable = []
initialfeatures = list(df.columns)

sf.ImportLossFunction(lossfunction, direction='ascend')
# import loss function handle and optimize direction, 'ascend' for AUC, ACC, 'descend' for logloss etc.
sf.InitialNonTrainableFeatures(notusable)  # those features that is not trainable in the dataframe, user_id, string, etc
sf.InitialFeatures(initialfeatures)  # initial initialfeatures as list
sf.GenerateCol()  # generate features for selection
sf.SetFeatureEachRound(50, False)
# set number of feature each round, and set how the features are selected from all features (True: sample selection, False: select chunk by chunk)
sf.clf = LogisticRegression()  # set the selected algorithm, can be any algorithm
# 结果记录到log中，最高得分在最后
sf.SetLogFile('logs/record.log')  # log file
sf.run(validation)
# run with validation function, validate is the function handle of the validation function, return best features combination
