import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import autokeras as ak
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
"""
keras模型预测

0~1 0
1~2 122
2~3 913
3~4 763
4~5 235
不均匀数据集导致回归预测高分区不准确
"""
directory = "/opt/data/SCUT-FBP5500_v2/Images/train/face/"
df_rates = pd.read_csv("/opt/data/SCUT-FBP5500_v2/All_Ratings.csv", header=None, names=['filename', 'score'])
df_rates = df_rates[df_rates['filename'].str.find("AF")>=0]
df_rates['score'] = df_rates['score'].astype(int)
df_rates_mean = df_rates.groupby('filename').mean()
df_rates_mean.reset_index(inplace=True)
# 挑选测试用例
try:
    score0 = df_rates_mean[df_rates_mean['score']<1].head(2)
except Exception:
    score0 = None
try:
    score1 = df_rates_mean[df_rates_mean['score'].between(1,2)].head(2)
except Exception:
    score1 = None
try:
    score2 = df_rates_mean[df_rates_mean['score'].between(2,3)].head(2)
except Exception:
    score2 = None
try:
    score3 = df_rates_mean[df_rates_mean['score'].between(3,4)].head(2)
except Exception:
    score3 = None
try:
    score4 = df_rates_mean[df_rates_mean['score'].between(4,5)].head(2)
except Exception:
    score4 = None

# model = load_model("model_beauty_v1", custom_objects=ak.CUSTOM_OBJECTS)
model = load_model("model_beauty_v1")


def path_to_image(path, image_size, num_channels, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=num_channels, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], num_channels))
    return img


def predict(image_path):
    global model
    image = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", interpolation="bilinear")
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # input_arr = path_to_image(image_path, (350, 350), 3, "bilinear")
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    predictions = model.predict(tf.expand_dims(input_arr, -1))
    return predictions[0][0]


def test_score(se):
    global directory
    image_path = directory + se["filename"]
    preds = predict(image_path)
    print("test:", se['filename'], "lable:", se['score'], "predict:", preds)
    return preds


if __name__ == "__main__":
    test_scores = pd.concat([score0, score1, score2, score3, score4])
    test_scores.apply(test_score, axis=1)
