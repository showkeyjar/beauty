import csv
import cv2
import numpy as np
import pandas as pd
"""
转换 SCUT-FBP55000_v2 数据集到csv格式
参考：
https://bbs.huaweicloud.com/blogs/detail/278704
https://github.com/spytensor/prepare_detection_dataset

pip install opencv-python
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel

/path/to/image,xmin,ymin,xmax,ymax,label
/mfs/dataset/face/0d4c5e4f-fc3c-4d5a-906c-105.jpg,450,154,754,341,face
"""

face_cascade = cv2.CascadeClassifier("haarcascade_fontalface_default.xml")
prototxt_path = "/opt/opencv/deploy.prototxt"
model_path = "/opt/opencv/res10_300x300_ssd_iter_140000_fp16.caffemodel"
cv_model = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

image_path = "/opt/data/SCUT-FBP5500_v2/Images/train/face/"
df_label = pd.read_csv("/opt/data/SCUT-FBP5500_v2/All_Ratings.csv")
df_label = df_label.groupby("Filename").mean()
# 先仅处理女性（女性和男性的美差异较大）
df_label = df_label[df_label["Filename"].str.find("F")>=0]


def gen_label(se):
    global image_path, cv_model
    image = cv2.imread(image_path + str(se["Filename"]))
    h, w = image.shape[:2]
    start_x=start_y=0
    end_x = w
    end_y = h
    # 注意：这里 104.0, 177.0, 123.0 表示b-104，g-177,r-123
    # 这里实际应减去数人脸据集的图像均值而不是当前图像均值来对图像进行归一化
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),(104.0, 177.0, 123.0))
    cv_model.setInput(blob)
    output = np.squeeze(cv_model.forward())
    if len(output)>0:
        # 选取最大置信度的结果
        max_box = output[0, 3:7] * np.array([w, h, w, h])
        max_confidence = output[0, 2]
        for i in range(0, output.shape[0]):
            confidence = output[i, 2]
            if confidence > max_confidence:
                max_box = output[i, 3:7] * np.array([w, h, w, h])
                max_confidence = confidence
        start_x, start_y, end_x, end_y = max_box.astype(int)
    line_str = image_path + str(se["Filename"]) + "," + str(start_x) + "," + str(start_y) + "," + str(end_x) + "," + str(end_y) + "," + str(se["Rating"])
    return line_str


if __name__=="__main__":
    df_label["csv_str"] = df_label.apply(gen_label, axis=1)
    df_label["csv_str"].drop_duplicates().to_csv("/opt/data/SCUT-FBP5500_v2/Images/train/labels.csv", index=False, header=False, quoting=csv.QUOTE_NONE)
    print("gen label finished.")
