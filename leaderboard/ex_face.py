import os
import cv2
import glob
import time
import shutil
import pandas as pd
from pathos.multiprocessing import ProcessingPool as Pool
from feature.tools import get_seg_face, single_face_alignment, cut_face_img
"""
抽取人脸
# imagededup 依赖 tensorflow, 但pip在windows下无法正常安装tf
conda install tensorflow
pip install opencv-python dill 
# conda install cmake
# 先安装最新vstudio c++ (太新的vc++也会编译失败，使用conda安装dlib)
conda install -c conda-forge dlib
# conda install Pillow
pip install imagededup
# imagededup 指定的numpy等包与pandas不兼容，安装完后再升级
pip install -U numpy
pip install pathos
pip install -U PyWavelets
pip install tables pandas scikit-image

python -m leaderboard.ex_face
"""
record_file = "data/face_record.ftr"
try:
    df_record = pd.read_feather(record_file)
except Exception as e:
    df_record = None
    print(e)


def gen_face(img_path):
    se = None
    img = None
    image_bgra, landmarks = get_seg_face(img_path)
    rot_img = single_face_alignment(image_bgra, landmarks)
    if rot_img is not None:
        img = cut_face_img(rot_img)
    if img is not None:
        # 这里不能再用uuid，而需要用时间戳
        # img_id = uuid.uuid4()
        img_id = str(round(time.time()*1000))
        new_path = "data/face/" + str(img_id) + ".jpg"
        cv2.imwrite(new_path, img)
        se = pd.Series({"img_id": img_id, "img_path": new_path, "crawl_path": img_path})
    return se


def get_face(img_file):
    img_file = img_file.replace("\\", "/")
    se = gen_face(img_file)
    print("process file:", img_file)
    return se


if __name__=="__main__":
    try:
        shutil.rmtree("data/face")
    except Exception as e:
        print(e)
    try:
        os.makedirs("data/face")
    except Exception as e:
        pass
    image_files = glob.glob("leaderboard/AutoCrawler/download/*/*.jpg")
    # 排除已经处理过的文件
    if df_record is not None:
        try:
            finish_files = df_record['crawl_path'].tolist()
            image_files = [f for f in image_files if f not in finish_files]
        except Exception as e:
            print(e)
    # 单线程
    # list_record = []
    # for img_file in image_files:
    #     img_file = img_file.replace("\\", "/")
    #     se = gen_face(img_file)
    #     list_record.append(se)
    # 多线程方式
    with Pool(16) as pool:
        list_record = pool.map(get_face, image_files)
    list_record = [se for se in list_record if se is not None]
    df_record = pd.DataFrame(list_record)
    df_record.to_feather(record_file)
    print("write faces:", len(image_files))
    print("gen face finished.")
