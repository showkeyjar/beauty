import os
import argparse
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import image

"""
人脸评分
the next 3 lines of code are for my machine and setup due to https://github.com/tensorflow/tensorflow/issues/43174
"""
record_file = "../data/face_class_record.ftr"
try:
    df_record = pd.read_feather(record_file)
except Exception as e:
    df_record = None
    print(e)

model = keras.models.load_model('../data/models/ef_v2_model.h5', compile=False)


def load_img(path):
    if not os.path.exists(path):
        return None
    img = image.load_img(path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    # 注意 train ImageDataGenerator 对图像做了 rescale，所以这里对应的同样需要归一化
    img_array /= 255.0
    img_batch = np.expand_dims(img_array, axis=0)
    return img_batch


def predict(image_path):
    global model
    input_arr = load_img(image_path)
    if input_arr is None:
        return None
    predictions = model.predict(input_arr)
    return predictions[0][0]


def score_face(se):
    global directory
    image_path = "../" + se["img_path"]
    preds = predict(image_path)
    print(image_path, "score:", preds)
    return preds


if __name__ == "__main__":
    # 预测结果
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=int)
    args = parser.parse_args()
    force = False
    if args.f:
        force = True
    if df_record is not None and "score" not in df_record.columns:
        force = True
    if force:
        df_record["score"] = df_record.apply(score_face, axis=1)
    else:
        df_record.loc[df_record['score'].isna(), "score"] = df_record[df_record['score'].isna()].apply(score_face, axis=1)
    df_record.to_feather(record_file)
    print("finish score new data")
