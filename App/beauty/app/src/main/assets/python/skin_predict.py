import io
import cv2
import base64
import numpy as np
from PIL import Image
from joblib import load
from os.path import dirname, join
from part_dlib import get_face_parts
"""
预测皮肤类型
分两步预测
1. 使用dlib分割出区域   python完成
2. solo-learn backbones预测出特征
    pytorch model -> onnx -> tflite 
3. lda预测出分类        python完成
    python model
"""
model_file = join(dirname(__file__), "lda_skin.jb")
lda = load(model_file)


def cut_cheek(str_data):
    decode_data = base64.b64decode(str_data)
    np_data = np.fromstring(decode_data, np.uint8)
    image = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    part_img = get_face_parts(image, "inside")
    cheek = None
    if "left_cheek" in part_img:
        cheek = part_img["left_cheek"]
    elif "right_cheek" in part_img:
        cheek = part_img["right_cheek"]
    if cheek is not None:
        cheek = cv2.resize(cheek, (80, 96), interpolation=cv2.INTER_LINEAR)
        # cheek = cheek.astype(np.float32)
        pil_img = Image.fromarray(cheek)
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        return buff.getvalue()
    return None


def predict(jarray):
    global lda
    np_data = np.array(jarray)
    result = lda.transform([np_data])
    class_index = np.argmax(result[0])
    return class_index.item()


if __name__=="__main__":
    # df_data = pd.read_feather("../../data/cheek_boyl.ftr")
    train_x = np.array([1.0, 0.9, 0.4, 0.14, 0.5])
    result = predict(train_x[0])
    print(result)
