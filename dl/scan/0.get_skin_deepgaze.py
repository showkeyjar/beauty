import glob
import numpy as np
import cv2
from deepgaze.color_detection import RangeColorDetector

"""
准备数据

使用deepgaze切出所有皮肤
deepgaze切皮肤有一个严重的弊端：
我们切出人脸皮肤的目的就是为了评测，但deepgaze通过颜色来区分出皮肤的部分
刚好把皮肤的缺陷都剔除掉了，但这些缺陷正是我们需要评测的部分
所以不能使用deepgaze进行皮肤提取

git clone git://github.com/mpatacchiola/deepgaze.git
cd deepgaze
git checkout 2.0
python setup.py install
"""


file_format = "/opt/data/SCUT-FBP5500_v2/Images/train/face/AF*.jpg"
af_files = glob.glob(file_format)
for img_path in af_files:
    #适用于亚洲女性的HSV范围：色调 饱和度 明度
    min_range = np.array([0, 16, 32], dtype = "uint8") #lower HSV boundary of skin color
    max_range = np.array([50, 255, 255], dtype = "uint8") #upper HSV boundary of skin color
    my_skin_detector = RangeColorDetector(min_range, max_range) #Define the detector object
    image = cv2.imread(img_path) #Read the image with OpenCV
    #We do not need to remove noise from this image so morph_opening and blur are se to False
    image_filtered = my_skin_detector.returnFiltered(image, morph_opening=False, blur=False, kernel_size=3, iterations=1)
    file_name = img_path.split("/")[-1]
    cv2.imwrite("/opt/data/SCUT-FBP5500_v2/skins/" + file_name, image_filtered) #Save the filtered image
    print("finish file:", img_path)
