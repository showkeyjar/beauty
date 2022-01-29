import io
import cv2
import base64
import numpy as np
from PIL import Image
from os.path import dirname, join
import tensorflow as tf
from alibi.explainers import AnchorImage

"""
模型解释(改用 alibi )
pip install alibi
alibi 没有针对image regression问题的解释逻辑
"""

# Load TFLite model and allocate tensors.
model_file = join(dirname(__file__), "model_beauty_q_v2.tflite")
interpreter = tf.compat.v1.lite.Interpreter(model_path=model_file)


def predict_batch(inp):
    """
    参考: https://www.kaggle.com/grafael/fast-predictions-tflite-1h-3x-faster
    """
    global interpreter
    interpreter.allocate_tensors()
    input_det = interpreter.get_input_details()[0]
    output_det = interpreter.get_output_details()[0]
    input_index = input_det["index"]
    output_index = output_det["index"]
    input_shape = input_det["shape"]
    output_shape = output_det["shape"]
    input_dtype = input_det["dtype"]
    output_dtype = output_det["dtype"]
    inp = inp.astype(input_dtype)
    count = inp.shape[0]
    out = np.zeros((count, output_shape[1]), dtype=output_dtype)
    for i in range(count):
        interpreter.set_tensor(input_index, inp[i:i+1])
        interpreter.invoke()
        out[i] = interpreter.get_tensor(output_index)[0]
    return out


def predict(img):
    """
    参考:https://github.com/tensorflow/tensorflow/issues/37012
    """
    global interpreter
    batch_input = np.vstack(img)
    print("batch_input shape:", batch_input.shape)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.resize_tensor_input(input_details[0]['index'], batch_input.shape)
    interpreter.resize_tensor_input(output_details[0]['index'], batch_input.shape)
    # output_shape = output_details[1]['shape']
    # interpreter.resize_tensor_input(output_details[1]['index'], (batch_input.shape[0], output_shape[1], output_shape[2], output_shape[3]))
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_details[0]['index'], batch_input)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data


predict_fn = lambda x: predict_batch(x)


def explain_img(old_img):
    """
    参考: https://docs.seldon.io/projects/alibi/en/latest/examples/anchor_image_imagenet.html
    """
    img = cv2.resize(old_img, (300, 300), interpolation=cv2.INTER_NEAREST)
    img = img.astype(np.float32)
    img /= 255.0
    print("get img shape:", img.shape, img.dtype)
    segmentation_fn = 'slic'
    kwargs = {'n_segments': 15, 'compactness': 20, 'sigma': .5}
    explainer = AnchorImage(predict_batch, (300, 300, 3), segmentation_fn=segmentation_fn,
                            segmentation_kwargs=kwargs, images_background=None)
    explanation = explainer.explain(img, threshold=.95, p_sample=.5, tau=0.25)
    pil_img = Image.fromarray(explanation.anchor)
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return buff.getvalue()


def gen_result(str_data):
    decode_data = base64.b64decode(str_data)
    np_data = np.fromstring(decode_data, np.uint8)
    old_img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    explain_result = explain_img(old_img)
    return explain_result


if __name__=="__main__":
    # python -X faulthandler report_lite.py
    img = cv2.imread("/opt/data/SCUT-FBP5500_v2/Images/train/face/AF1031.jpg")
    grid = explain_img(img)
    # explainer.save(grid, ".", "occ_sens_lite.png")
    grid.save('occ_sens_lite.png')
