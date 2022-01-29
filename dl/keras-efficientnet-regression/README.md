# Transfer Learning with EfficientNet for Image Regression in Keras - Using Custom Data in Keras

![teaser](./teaser.png)

This is the Repo for my recent blog post: [Transfer Learning with EfficientNet for Image Regression in Keras - Using Custom Data in Keras](https://rosenfelder.ai/keras-regression-efficient-net/)

There are hundreds of tutorials online available on how to use Keras for deep learning. But at least to my impression, 99% of them just use the MNIST dataset and some form of a small custom convolutional neural network or ResNet for classification. Personally, I dislike the general idea of always using the easiest dataset for machine learning and deep learning tutorials since this leaves many important questions unanswered. Adapting these tutorials to a custom dataset for a regression problem can be a daunting and time-consuming task with hours of Googling and reading old StackOverflow questions or the official Keras documentation. Through this tutorial, I want to show you how to use a custom dataset and use transfer learning to get great results with very little training time. The following topics will be part of this tutorial:

- use ImageDataGenerators and Pandas DataFrames to load your custom dataset
- augment your image to improve prediction results
- plot augmentations
- adapt the state-of-the-art EfficientNet to a regression
- use the new Ranger optimizer from `tensorflow_addons`
- compare the EfficientNet results to a simpler custom convolutional neural network

## 颜值分析

颜值分析使用 tf-explain + Chaquopy 
pip install tf-explain

https://chaquo.com/chaquopy/doc/current/android.html

人脸区域分割：face-parsing.PyTorch + torch2tflite

git clone git://github.com/zllrunning/face-parsing.PyTorch.git

实际测试发现：人脸区域类别名称推断并不准确，改用dlib识别人脸的68个关键点再分割
dlib区域segment

"center":
{
    "left_eyebrow": [17, 18, 19, 20, 21],
    "right_eyebrow": [22, 23, 24, 25, 26],
    "left_upper_eyelid": (36, 37, 38, 39),
    "right_upper_eyelid": (42, 43, 44, 45),
    "left_lower_eyelid": (36, 41, 40, 39),
    "right_lower_eyelid": (42, 47, 46, 45),
    "nose_bridge": [27, 28, 29, 30],
},
"inside":
{
    "left_eye": [36, 37, 38, 39, 40, 41],
    "right_eye": [42, 43, 44, 45, 46, 47],
    "nose_tip": [30, 31, 32, 33, 34, 35],
    "mouse": [60, 61, 62, 63, 64, 65, 66, 67],
    "upper_lip": [48, 49, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64],
    "lower_lip": [60, 67, 66, 65, 64, 48, 59, 58, 57, 56, 55, 54],
    "jaw": [48, 59, 58, 57, 56, 55, 54, 5, 6, 7, 8, 9, 10, 11],
    "left_cheek": [0, 36, 41, 40, 39, 1, 2, 3, 4, 48, 31],
    "right_cheek": [42, 47, 46, 45, 16, 35, 54, 12, 13, 14, 15],
},
"top":
{
    "forehead": [0, 17, 18, 19, 20, 21, 27, 22, 23, 24, 25, 26, 16],
}


pip3 install --index-url https://google-coral.github.io/py-repo/ tflite_runtime

python3 converter.py --torch-path /home/meteo/models/face_parse.pth --tflite-path face_part.tflite --sample-file 1.png --target-shape 512 512 3

### 流程：

1.在生成颜值分析的同时，使用face-parsing对人脸进行分割

2.然后统计颜色过白的区域

3.根据颜色太白的区域查找知识库，得到解决方案

4.使用Room存储sqlite知识库
    参考：https://developer.android.com/training/data-storage/room

5.增加对皮肤的建模识别(使用聚类的方式)
    皮肤的保养是重点

6.使用图像聚类得到皮肤样本
    参考：https://github.com/wvangansbeke/Unsupervised-Classification

### todo

1.人脸部分得分的改进办法：分别计算4-5分样本各区域的置信度，然后与现有置信度比较
