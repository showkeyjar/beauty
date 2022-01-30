import io
import cv2
import math
import base64
import numpy as np
from PIL import Image
from os.path import dirname, join
import tensorflow as tf
from explain_lite import OcclusionSensitivity

"""
模型解释
"""
model_file = join(dirname(__file__), "model_beauty_q_v2.tflite")
interpreter = tf.compat.v1.lite.Interpreter(model_path=model_file)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# interpreter.allocate_tensors()


def gen_result(str_data):
    try:
        decode_data = base64.b64decode(str_data)
        np_data = np.fromstring(decode_data, np.uint8)
        old_img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
        img = cv2.resize(old_img, (300, 300), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)
        img /= 255.0
        print(img.shape, img.dtype)
        img_width = img.shape[0]
        img_height = img.shape[1]
        data = ([img], None)
        # Start explainer
        explainer = OcclusionSensitivity()
        # patch_size 是指分析的间隔，间隔越小越准确，越大速度越快
        # 实际测试patch_size = patch_length只推理1次，推理速度反而变慢，并伴随有光栅
        patch_length = max(img_width, img_height)
        patch_size = math.floor(patch_length / 100)
        grid = explainer.explain(data, interpreter, class_index=0, patch_size=patch_size)  # 0 is regression class index in train
        print(grid.shape, grid.dtype)
        pil_img = Image.fromarray(grid)
        buff = io.BytesIO()
        pil_img.save(buff, format="PNG")
        return buff.getvalue()
    except Exception as e:
        print(e)
    return None


if __name__=="__main__":
    # python -X faulthandler report_lite.py
    # Load a sample image (or multiple ones)
    img_width = img_height = 300
    img = tf.keras.preprocessing.image.load_img("/opt/data/SCUT-FBP5500_v2/Images/train/face/AF1031.jpg", target_size=(300, 300))
    img = tf.keras.preprocessing.image.img_to_array(img)
    data = ([img], None)
    # Start explainer
    explainer = OcclusionSensitivity()
    # patch_size 是指分析的间隔，间隔越小越准确，越大速度越快
    patch_size = math.floor(img_width / 5)
    grid = explainer.explain(data, interpreter, class_index=0, patch_size=patch_size)  # 0 is regression class index in train
    explainer.save(grid, ".", "occ_sens_lite.png")
