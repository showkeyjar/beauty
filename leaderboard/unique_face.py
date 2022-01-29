import os
import shutil
import logging
import numpy as np
import pandas as pd
from imagededup.methods import PHash

"""
删除重复图片
"""
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(format="%(thread)d %(asctime)s %(name)s:%(levelname)s:%(lineno)d:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

unique_record_file = "../data/face_class_record.ftr"
try:
    df_record = pd.read_feather(unique_record_file)
except Exception as e:
    df_record = None
    print(e)


def dict_to_set(result_dict):
    """
    重新组织 imagededup 生成的结果，并去重
    """
    ret_list = set()
    result_list = [v + [k] for k, v in result_dict.items()]
    for r_list in result_list:
        if r_list is not None:
            r_list.sort()
            ret_list.add(tuple(r_list))
    # 对结果本身去重，主要由于 imagededup 丑陋的结果组织形式
    logger.debug("get unique img list:" + str(len(ret_list)))
    return ret_list


def remove_duplicates():
    global df_record, unique_record_file
    phasher = PHash()
    # 生成图像目录中所有图像的二值hash编码
    encodings = phasher.encode_images(image_dir='Face-Pose-Net/output_render/')
    # 对已编码图像寻找重复图像
    duplicates = phasher.find_duplicates(encoding_map=encodings)
    logger.debug("find duplicates:" + str(len(duplicates)))
    print(duplicates[list(duplicates.keys())[0]])
    dup_list = dict_to_set(duplicates)
    new_img = 0
    # 删除除第1条记录以外的其他文件
    # 第1条要与知识库建立联系，不能随便删除
    for dup in dup_list[1:]:
        try:
            os.remove("data/unique_face/" + dup[0])
            logger.debug("remove file " + dup[0])
        except Exception as e:
            logger.warning(e)
        # 检查是否已收录
        img_id = dup[0].replace(".jpg", "")
        se = pd.Series({"img_path": "data/unique_face/" + dup[0], "img_id": img_id})
        if df_record is not None:
            df_exist_record = df_record[df_record['img_id']==img_id]
            if len(df_exist_record) == 0:
                df_record = df_record.append(se, ignore_index=True)
                new_img = new_img + 1
        else:
            df_record = pd.DataFrame([se])
            new_img = new_img + 1
    try:
        df_record.drop('level_0', axis=1, inplace=True)
    except Exception as e:
        logger.warning(e)
    try:
        logger.debug("select unique images: " + str(new_img))
        df_record.dropna(subset=['img_path'], inplace=True)
        df_record.reset_index(inplace=True)
        df_record.to_feather(unique_record_file)
    except Exception as e:
        logger.exception(e)


if __name__=="__main__":
    # 去除重复face
    remove_duplicates()
    print("remove duplicates finish")
