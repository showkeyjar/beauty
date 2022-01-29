import os
import bisect
import shutil
import argparse
import pandas as pd
from deepface import DeepFace

"""
按年龄区间划分
年龄维度：[0-8],[8-16],[16-24],[24-32],[32-40],[40-48],[48-56],[56-64],[64-]
按角度划分
角度区间：[00, 22, 40, 55, 75]

pip install deepface

error: Support for codec 'lz4' not build
mamba uninstall pyarrow
mamba install -c anaconda lz4
pip install pyarrow
mamba install pandas
"""
age_range = [8, 16, 24, 32, 40, 48, 56, 64]
pose_range = ["00", "22", "40", "55", "75"]

record_file = "../data/unique_face_record.ftr"
record_class_file = "../data/face_class_record.ftr"
df_record = None
try:
    df_record = pd.read_feather(record_file)
    print("load df_record:",df_record.columns)
except Exception as e:
    print(e)


def get_face_info(se):
    global age_range, pose_range
    # new_img_path = se['img_path']
    # img_path = "../" + se['img_path'].replace("\\", "/")
    # img_name = img_path.split("/")[-1]
    list_pose = []
    img_name = se['img_id']
    for pose in pose_range:
        new_path = None
        emotion = None
        gender = None
        race = None
        age_index = None
        img_path = "Face-Pose-Net/output_render/" + img_name + "/" + img_name + "_rendered_aug_-" + pose + "_00_10.jpg"
        if not os.path.exists(img_path):
            print(img_path, "not exists")
            continue
        try:
            obj = DeepFace.analyze(img_path = img_path, actions = ['age', 'gender', 'race', 'emotion'], enforce_detection=False)
            # 先只考虑表情和年龄维度
            age_index = bisect.bisect_left(age_range, obj["age"])
            gender = obj["gender"]
            race = obj["dominant_race"]
            emotion = obj["dominant_emotion"]
            new_path = "../data/class/" + pose + "/" + emotion + "/" + str(age_index) + "/"
        except Exception as e:
            print(e)
        if new_path is not None:
            try:
                os.makedirs(new_path)
            except Exception:
                pass
            try:
                shutil.copyfile(img_path, new_path + img_name + ".jpg")
            except Exception as e:
                print(e)
            print("process file:", new_path + img_name + ".jpg")
            new_img_path = new_path.replace("../", "") + img_name + ".jpg"
            se = pd.Series({'emotion': emotion, 'age': age_index, 'gender': gender, 'race': race, 'img_path': new_img_path})
            list_pose.append(se)
    df_result = None
    if len(list_pose)>0:
        df_result = pd.DataFrame(list_pose)
    return df_result


if __name__=="__main__":
    # 生成人脸属性
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=int)
    args = parser.parse_args()
    force = False
    if args.f:
        force = True
    if force or "gen_class" not in df_record.columns:
        list_new_record = df_record.apply(get_face_info, axis=1)
    else:
        list_new_record = df_record[~df_record['gen_class']==True].apply(get_face_info, axis=1)
        if os.path.exists(record_class_file):
            df_old = pd.read_feather(record_class_file)
            list_new_record.append(df_old)
    list_new_record = [df for df in list_new_record if df is not None]
    df_new_record = pd.concat(list_new_record, ignore_index=True)
    df_new_record.to_feather(record_class_file)
    df_record['gen_class'] = True
    df_record.to_feather(record_file)
    print("face age finished")
