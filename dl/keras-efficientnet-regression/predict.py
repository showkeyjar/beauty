from tensorflow.keras import layers, Model
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd

# the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
import tensorflow as tf

directory = "/opt/data/SCUT-FBP5500_v2/Images/train/face/"
df_rates = pd.read_csv("/opt/data/SCUT-FBP5500_v2/All_Ratings.csv", header=None, names=['filename', 'score'])
df_rates = df_rates[df_rates['filename'].str.find("AF")>=0]
df_rates['score'] = df_rates['score'].astype(int)
df_rates_mean = df_rates.groupby('filename').mean()
df_rates_mean.reset_index(inplace=True)
# 挑选测试用例
head_num = 5
try:
    score0 = df_rates_mean[df_rates_mean['score']<1]
    print("0~1", len(score0))
    score0 = score0.head(head_num)
except Exception:
    score0 = None
try:
    score1 = df_rates_mean[df_rates_mean['score'].between(1, 2)]
    print("1~2", len(score1))
    score1 = score1.head(head_num)
except Exception:
    score1 = None
try:
    score2 = df_rates_mean[df_rates_mean['score'].between(2, 3)]
    print("2~3", len(score2))
    score2 = score2.head(head_num)
except Exception:
    score2 = None
try:
    score3 = df_rates_mean[df_rates_mean['score'].between(3, 4)]
    print("3~4", len(score3))
    score3 = score3.head(head_num)
except Exception:
    score3 = None
try:
    score4 = df_rates_mean[df_rates_mean['score'].between(4, 5)]
    print("4~5", len(score4))
    score4 = score4.head(head_num)
except Exception:
    score4 = None


def adapt_efficient_net() -> Model:
    """This code uses adapts the most up-to-date version of EfficientNet with NoisyStudent weights to a regression
    problem. Most of this code is adapted from the official keras documentation.

    Returns
    -------
    Model
        The keras model.
    """
    pre_model = keras.models.load_model('pretrain_models/efficientnetv2-s-imagenet.h5')
    # Freeze the pretrained weights
    # pre_model.trainable = False
    inputs = pre_model.input
    x = layers.BatchNormalization()(pre_model.output)
    outputs = layers.Dense(1, name="pred")(x)
    # Compile
    model = keras.Model(inputs, outputs, name="EfficientNetV2")
    return model


# model = adapt_efficient_net()
# model.load_weights("./data/models/eff_net.h5")
model = keras.models.load_model('ef_v2_model.h5', compile=False)


def load_img(path):
    img = image.load_img(path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    # 注意 train ImageDataGenerator 对图像做了 rescale，所以这里对应的同样需要归一化
    img_array /= 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def predict(image_path):
    global model
    # image = tf.keras.preprocessing.image.load_img(image_path, color_mode="rgb", interpolation="bilinear")
    # input_arr = tf.keras.preprocessing.image.img_to_array(image)
    # input_arr = path_to_image(image_path, (224, 224), 3, "bilinear")
    # input_arr = np.array([input_arr])  # Convert single image to a batch.
    # predictions = model.predict(tf.expand_dims(input_arr, -1))
    input_arr = load_img(image_path)
    predictions = model.predict(input_arr)
    return predictions[0][0]
    # return predictions


def score_loss(y_true, y_pred):
    """
    由于数据集分布不平衡，自定义loss
    """
    abs_score = tf.abs(y_true - y_pred)
    # 超过 float32 大数导致 nan
    # sq_score = tf.math.square(abs_score)
    # 由于极值总是不准确，加大极值的惩罚力度
    mult_score = tf.math.multiply(tf.abs(y_true - 3), 2) + 1
    new_score = tf.math.multiply(abs_score, mult_score)
    # 分值越高越重视(惩罚力度越大)
    beauty_loss = tf.reduce_mean(new_score)
    return beauty_loss.numpy()


def test_score(se):
    global directory
    image_path = directory + se["filename"]
    preds = predict(image_path)
    loss = score_loss(np.array(se['score'], dtype=np.float64), np.array(preds, dtype=np.float64))
    print("test:", se['filename'], "lable:", round(se['score'], 2), "predict:", round(preds, 2), "loss:", round(loss, 2))
    return preds


if __name__ == "__main__":
    # 检查预测较差的样本 
    # /opt/data/SCUT-FBP5500_v2/Images/train/face/AF1031.jpg 偏低
    # /opt/data/SCUT-FBP5500_v2/Images/train/face/AF1025.jpg 偏高
    test_scores = pd.concat([score0, score1, score2, score3, score4])
    test_scores.apply(test_score, axis=1)

