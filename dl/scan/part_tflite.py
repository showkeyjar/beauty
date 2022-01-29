#!/usr/bin/python
# -*- encoding: utf-8 -*-
import numpy as np
import cv2
from os.path import dirname, join
import tensorflow as tf

"""
判断人脸部位
"""

# Load TFLite model and allocate tensors.
model_file = join(dirname(__file__), "face_part.tflite")
interpreter = tf.compat.v1.lite.Interpreter(model_path=model_file)

# Get input and output tensors.
input_details = interpreter.get_input_details()
print("input:", input_details)
output_details = interpreter.get_output_details()
print("output:", output_details)


def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path='parsing_face.jpg'):
    # Colors for all 19 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    # part_names = ['hair', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r', 'nose', 'mouth', 'skin', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'bg', 'hat']
    part_names = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat', 'bg']
    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        print("cls ", pi," find shape:", index[0].shape, index[1].shape)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    # 确保文字在最上层
    for pi in range(0, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        try:
            cv2.putText(vis_parsing_anno_color, str(pi) + ":" + part_names[pi], (index[0][0] + 1,index[1][0] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        except Exception as e:
            print(e)

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    if save_im:
        # cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
        cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    # return vis_im


def evaluate(img_path='./data'):
    global interpreter
    image = cv2.imread(img_path)
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
    img = image / 255
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    img = (img - mean) / std
    img = img.astype(np.float32)
    # change to channel first
    img = np.moveaxis(img, 2, 0)
    print(img.shape, img.dtype)
    interpreter.allocate_tensors()
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], [img])
    interpreter.invoke()
    preds = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
    parsing = preds[0].argmax(0)
    print(np.unique(parsing))
    vis_parsing_maps(image, parsing, stride=1, save_im=True)


if __name__ == "__main__":
    evaluate(img_path='1.png')
