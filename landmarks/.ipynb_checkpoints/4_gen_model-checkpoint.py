# %% coding=utf-8
import pandas as pd
from tpot import TPOTClassifier, TPOTRegressor
from sklearn.model_selection import train_test_split

beauty_data = pd.read_csv('data/face/df_input.csv', nrows=1000)

with open('logs/result_cols.txt', 'r') as f:
    select_cols = str(f.readline())

select_cols = select_cols.split(' ')

beauty_data = beauty_data[select_cols]
print(beauty_data.shape)
try:
    beauty_data.drop(['file'], axis=1, inplace=True)
except:
    pass

# 将评分取整，便于实现多分类问题
beauty_data['Rating'] = beauty_data['Rating'].round().astype('int32')

X_train, X_test, y_train, y_test = train_test_split(beauty_data.drop('Rating', axis=1), beauty_data['Rating'],
                                                    train_size=0.75, test_size=0.25)
print('split train test data')
tpot = TPOTRegressor(scoring='r2', n_jobs=-1, early_stop=5, verbosity=2)
# 单线程运行缓慢，增加参数 n_jobs=-1, use_dask=True
# tpot = TPOTClassifier(generations=5, population_size=30, n_jobs=-1, verbosity=2, random_state=42, use_dask=True)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('model/tpot_beauty_pipeline.py')
