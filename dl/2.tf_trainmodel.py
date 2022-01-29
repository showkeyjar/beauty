import os
import cv2
import dill
import tensorflow
import autokeras
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model

# from tensorflow.python.framework import ops
"""
训练模型
"""

# 加载数据
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


def imageLoader(files, batch_size):
    L = len(files)
    # this line is just to make the generator infinite, keras needs that
    while True:
        batch_start = 0
        batch_end = batch_size
        while batch_start < L:
            limit = min(batch_end, L)
            X = np.array(files[batch_start:limit]['file'].apply(read_img).to_list())
            Y = np.array(files[batch_start:limit]['Rating'].to_list())
            yield (X, Y)  # a tuple with two numpy arrays with batch_size samples
            batch_start += batch_size
            batch_end += batch_size


# X_train = np.array(df_train['image_path'].apply(read_train_img).to_list())
# Y_train = df_train['labels'].array

custom_obj = autokeras.CUSTOM_OBJECTS
custom_obj['Normalization'] = tensorflow.keras.layers.experimental.preprocessing.Normalization
# custom_obj['Conv2D'] = tensorflow.keras.layers.Conv2D
seque = load_model('../model/keras_model.h5', custom_objects=custom_obj)

batch_size = 12
# with open('../model/train_datagen.pkl', 'rb') as file:
#     train_datagen = dill.load(file)

train_tensorboard = tensorflow.keras.callbacks.TensorBoard(log_dir='../logs')

train_checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath="../model/tf_train_model.h5")

train_stop = tensorflow.keras.callbacks.EarlyStopping(
    # 是否有提升关注的指标
    monitor='val_loss',
    # 不再提升的阈值
    min_delta=1e-2,
    # 2个epoch没有提升就停止
    patience=2,
    verbose=1)

# train_datagen.fit(X_test[:2000], )
# ops.reset_default_graph()
# del seque.layers[2]
history = seque.fit_generator(imageLoader(df_data, batch_size), epochs=1000, verbose=1,
                              callbacks=[train_tensorboard, train_checkpoint, train_stop], initial_epoch=0)

# test_csv_dir = "G:/data/SCUT-FBP5500_v2/Images/"
# df_test = pd.read_csv(test_csv_dir)
#
# X_test = df_test['image_path'].apply(read_test_img).to_list()
# Y_test = df_test['labels'].array
#
# score = seque.evaluate(np.asarray(X_test), Y_test, verbose=0)
# print(score)

# 2020-03-28 09:49:59.155391: I tensorflow/core/profiler/lib/profiler_session.cc:225] Profiler session started.
#   36946/Unknown - 629238s 17s/step - loss: 107440074304.1912 - accuracy: 0.0027