import os
import cv2
# import dill
import glob
from part_dlib import get_face_parts

"""
准备数据

使用dlib切出所有皮肤(为简单起见，先只提取脸颊皮肤)
pip install dill
pip install boost
pip install cmake
pip install dlib
pip install opencv-python
"""
# gen_parts = ["nose_bridge", "nose_tip", "upper_lip", "lower_lip", "jaw", "left_cheek", "right_cheek", "forehead"]
# 由于这里只关注皮肤的肤质，所以训练时可以统一到同样的尺寸，让模型不必关注尺寸
gen_parts = {"left_cheek": (80, 96), "right_cheek": (80, 96)}

file_format = "/opt/data/SCUT-FBP5500_v2/Images/train/face/AF*.jpg"

if __name__=="__main__":
    # ps -ef|grep "python 0.get_skin_dlib" | awk '{print $2}' | xargs kill -9
    af_files = glob.glob(file_format)
    for img_path in af_files:
        image = cv2.imread(img_path) 
        # 目前只有inside做好了，先只使用inside
        part_img = get_face_parts(image, "inside")
        save_file = img_path.replace("/opt/data/SCUT-FBP5500_v2/Images/train/face/", "/mnt/nfs196/soft/skin/")
        for p, v in gen_parts.items():
            save_file1 = save_file.replace(".jpg", p + ".jpg")
            if not os.path.exists(save_file1) and p in part_img and part_img[p] is not None:
                pic = part_img[p]
                pic = cv2.resize(pic, v, interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(save_file1, pic)
                print("save part ", p, " file:", save_file1)
        # save_file = save_file.replace(".jpg", ".pkl")
        # with open(save_file, "wb") as f:
        #     dill.dump(part_img, f)
        # print("save file:", save_file)
