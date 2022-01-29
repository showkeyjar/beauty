# %% coding=utf-8
import os
import cv2
import dlib
import glob
import numpy as np
import pandas as pd
from feature.tools import face_correct
from skimage.transform import integral_image
from skimage.feature import local_binary_pattern
from skimage.feature import multiblock_lbp
from deepgaze.color_detection import BackProjectionColorDetector
from deepgaze.color_detection import RangeColorDetector
from multiprocessing import Pool

"""
准备数据
todo 需要增加人脸对齐逻辑
"""

# 0.加载模型
predictor_path = "model/shape_predictor_68_face_landmarks.dat"
faces_folder_path = "E:/data/SCUT-FBP5500_v2/Images/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

my_back_detector = BackProjectionColorDetector()  # Defining the deepgaze color detector object

min_range = np.array([0, 48, 70], dtype="uint8")  # lower HSV boundary of skin color
max_range = np.array([20, 150, 255], dtype="uint8")  # upper HSV boundary of skin color
my_skin_detector = RangeColorDetector(min_range, max_range)  # Define the detector object


# 1.获取人脸区域
def get_one_face(img_path):
    src_img = None
    face = None
    face_dets = None
    img_path = img_path.replace('\\', '/')
    image_name = img_path.split('/')[-1]
    new_path = 'E:/data/face/' + image_name
    new_path = face_correct(img_path, new_path=new_path)
    if new_path is not None:
        src_img = dlib.load_rgb_image(new_path)
        face_dets = detector(src_img, 1)
        print("Number of faces detected: {}".format(len(face_dets)))
    if face_dets is not None and len(face_dets) > 0:
        face = face_dets[0]
    return src_img, face


# 2.提取关键点数据
def get_landmarks(img, d=None):
    face_shape = {}
    if d is not None:
        #print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
        f_width = abs(d.right() - d.left())
        f_height = abs(d.bottom() - d.top())
        print('width:' + str(f_width) + ', height:' + str(f_height))
        # Get the landmarks/parts for the face in box d.
        shape = predictor(img, d)
        print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
        for i in range(0, 67):
            for j in range(i+1, 68):
                face_shape[str(i) + '_' + str(j) + '_x'] = abs(shape.part(i).x - shape.part(j).x)/f_width
                face_shape[str(i) + '_' + str(j) + '_y'] = abs(shape.part(i).y - shape.part(j).y)/f_height
                #print(str(i) + '_' + str(j))
    return face_shape


# 3.提取皮肤颜色 Histogram Backprojection algorithm
def color_extract(image, d=None):
    global my_back_detector
    images_stack = None
    if d is not None:
        template = image[d.left():d.top(), d.right():d.bottom()]  # Taking a subframe of the image
        my_back_detector.setTemplate(template)  # Set the template
        image_filtered = my_back_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=7, iterations=2)
        images_stack = np.hstack((image, image_filtered))  # The images are stack in order
    return images_stack


# 生成lbph特征
def lbph_extract(img, d=None, eps=1e-7):
    skin_hist = {}
    if d is not None:
        no_points = 24
        radius = 8
        d_top = d.top()
        d_bottom = d.bottom()
        d_left = d.left()
        d_right = d.right()
        cropped = img[d_top:d_bottom, d_left:d_right]
        im_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        (histogram, _) = np.histogram(lbp.ravel(),
                                      bins=np.arange(0, no_points + 3),
                                      range=(0, no_points + 2))
        # now we need to normalise the histogram so that the total sum=1
        histogram = histogram.astype("float")
        histogram /= (histogram.sum() + eps)
        for i, h in enumerate(histogram):
            skin_hist['skin_' + str(i)] = h
    return skin_hist


# 生成带语义的 MB-LBP 高级特征
# skimage 的 multiblock_lbp 需要预先定义九宫格的宽度和高度
# 对于多尺度的图片，预处理的工作量较大
# 选取左额头、右额头、左脸颊、右脸颊、鼻梁、左下巴、右下巴 六个区域作为MB-LBP样本
def semantic_lbp_extract(image, d=None):
    int_img = integral_image(image)
    # 额头：计算眉毛顶点(20,25)与眼睛顶点(38,45)的垂直距离，以眉毛顶点作为样本区域底部中点，选择计算样本
    # 脸颊：计算人脸边缘(3,15)与鼻翼(32,36)的水平距离1，计算眼底(42,47)与鼻翼(32,36)的垂直距离2，
    #   选取1，2中较小的将鼻翼作为样本区域底部顶点，选择计算样本
    lbp_code = multiblock_lbp(int_img, 0, 0, 3, 3)
    return lbp_code


# 4.提取皮肤 HSV range color detector
def skin_extract(image, d=None):
    global my_skin_detector
    if d is not None:
        # We do not need to remove noise from this image so morph_opening and blur are se to False
        image_filtered = my_skin_detector.returnFiltered(image, morph_opening=False, blur=False, kernel_size=3,
                                                         iterations=1)
        cv2.imwrite("tomb_rider_filtered.jpg", image_filtered)  # Save the filtered image
        # Second image boundaries
        min_range = np.array([0, 58, 50], dtype="uint8")  # lower HSV boundary of skin color
        max_range = np.array([30, 255, 255], dtype="uint8")  # upper HSV boundary of skin color
        image = cv2.imread("tomb_rider_2.jpg")  # Read the image with OpenCV
        my_skin_detector.setRange(min_range, max_range)  # Set the new range for the color detector object
        # For this image we use one iteration of the morph_opening and gaussian blur to clear the noise
        image_filtered = my_skin_detector.returnFiltered(image, morph_opening=True, blur=True, kernel_size=3, iterations=1)
        cv2.imwrite("tomb_rider_2_filtered.jpg", image_filtered)  # Save the filtered image


# 5.保存特征
def save_features(faces):
    faces.to_csv('data/face/features.csv')


def execute(f):
    print("Processing file: {}".format(f))
    features = {'image_index': f}
    src_img, face = get_one_face(f)
    landmarks = get_landmarks(src_img, face)
    # print('get landmarks:' + str(landmarks))
    features.update(landmarks)
    skin_hists = lbph_extract(src_img, face)
    # print('get skin hist:' + str(skin_hists))
    features.update(skin_hists)
    print('processed: ' + f)
    return pd.Series(features)


if __name__ == "__main__":
    shape_size = []
    pool = Pool(processes=32)
    for f in glob.glob(os.path.join(faces_folder_path, "AF*.jpg")):
        # execute(f)
        print(f)
        shape_size.append(pool.apply_async(execute, (f,)))
    pool.close()
    pool.join()
    faces = pd.DataFrame()
    for pool_data in shape_size:
        # 这里再使用get()方法可以获取返回值
        faces = faces.append(pool_data.get(), ignore_index=True)
    # faces = pd.DataFrame.from_dict(shape_size)
    save_features(faces)
    # AF1183 生成有问题
