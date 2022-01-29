import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak
from sklearn.model_selection import train_test_split

"""
使用autokeras生成模型

读取SCUT-FBP5500_v2数据集
使用前请自行下载数据集：
https://github.com/HCIILAB/SCUT-FBP5500-Database-Release
"""
data = pd.read_excel('E:/data/SCUT-FBP5500_v2/All_Ratings.xlsx', None)
df_asian = data['Asian_Female']
df_data = df_asian.groupby('Filename').agg(np.mean)
df_data['file'] = df_data.index.astype(str)


def read_img(x):
    img_path = "E:/data/SCUT-FBP5500_v2/Images/" + x
    # gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    # gray = gray.reshape(224, 224, 1)
    # img_data = image.load_img(img_path, target_size=(224, 224))
    # x = image.img_to_array(img_data)
    try:
        data_ts = cv2.imread(img_path)
        # data_tf = tf.convert_to_tensor(img_data, np.float64)
        # data_ts = np.array(data_tf)
    except Exception as e:
        print(img_path)
        print(e)
        data_ts = None
    return data_ts


df_data['X'] = df_data['file'].apply(read_img)
df_data = df_data.dropna().reset_index(drop=True)

# 受GPU资源限制，使用少量样本探索网络结构
X = np.array(df_data['X'].to_list())
Y = np.array(df_data['Rating'].to_list())

# X = df_data['X'].to_list()[:200]
# Y = df_data['Rating'].to_list()[:200]

# x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# todo 生成近似参数
gen_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath="../model/keras_checkpoint.h5")

clf = ak.ImageClassifier()
clf.fit(X, Y, batch_size=8, callbacks=[gen_checkpoint])
# results = clf.predict(x_test)

model = clf.export_model()
model.save('../model/keras_model.h5')
