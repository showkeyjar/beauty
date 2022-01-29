# coding=utf-8
import os
import glob
import shutil
import logging
import pandas as pd
from datetime import datetime
from distutils.dir_util import copy_tree
from jinja2 import Environment, FileSystemLoader

"""
根据结果生成排行

pip install -U Jinja2
"""
logging.root.setLevel(logging.NOTSET)
logging.basicConfig(format="%(thread)d %(asctime)s %(name)s:%(levelname)s:%(lineno)d:%(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger(__name__)

class_names = {'undefined': '默认', 'neutral': '自然', 'angry': '生气', 'disgust': '恶心', 'fear': '害怕', 'happy': '开心', 'sad': '伤心', 'surprise': '惊讶'}
age_range = ["0-8","9-16","17-24", "25-32", "33-40", "41-48", "49-56", "57-64", "64+"]
pose_range = {"00": "正面", "22": "微侧22°", "40": "微侧40°", "55": "微侧55°", "75": "侧面"}

record_file = "../data/face_class_record.ftr"
try:
    df_record = pd.read_feather(record_file)
    df_record = df_record[(df_record['gender']=='Woman') & (df_record['race']=='asian')]
    df_record['score'].fillna(-1, inplace=True)
    df_record['score'] = df_record['score'].astype(float).round(2)
    df_record['age'].fillna(3, inplace=True)
    df_record['age'] = df_record['age'].astype(int)
    df_record['name'] = "佚名"
    df_record['img_id'] = df_record['img_path'].apply(lambda x: x[x.rfind("/") + 1:].replace(".jpg", ""))
    # df_record['img_path'] = df_record['img_path'].apply(lambda x:x+".jpg")
except Exception as e:
    df_record = None
    logger.warning(e)

env = Environment(
    loader=FileSystemLoader("templates", encoding='utf-8')
)


def copy_file(img_path):
    old_path = img_path[:img_path.rfind("/")]
    img_id = img_path[img_path.rfind("/") + 1:].replace(".jpg", "")
    new_path = "dist/" + old_path
    try:
        os.makedirs(new_path)
    except Exception:
        pass
    try:
        shutil.copyfile("../" + img_path, "dist/" + img_path)
    except Exception as e:
        logger.warning(e)
    # todo 拷贝原始文件
    try:
        shutil.copyfile("../data/unique_face/" + img_id + ".jpg", "dist/images/detail/" + img_id + ".jpg")
        # os.makedirs("dist/images/detail/" + img_id + "/")
        shutil.copytree("Face-Pose-Net/output_render/" + img_id, "dist/images/detail/" + img_id)
    except Exception as e:
        logger.warning(e)


def get_rank_class(index):
    if index==1:
        rank_class = "first"
        rank_display = "<img src='./assets/gold.png' class='medal'>"
    elif index==2:
        rank_class = "second"
        rank_display = "<img src='./assets/silver.png' class='medal'>"
    elif index==3:
        rank_class = "third"
        rank_display = "<img src='./assets/bronze.png' class='medal'>"
    else:
        rank_class = ""
        rank_display = str(index)
    return pd.Series({"rank_class": rank_class, "rank_display": rank_display})


def prepare_resource():
    try:
        shutil.rmtree("dist")
    except Exception as e:
        logger.warning(e)
    try:
        os.mkdir("dist")
    except Exception as e:
        logger.warning(e)
    try:
        os.makedirs("dist/detail")
    except Exception as e:
        logger.warning(e)
    try:
        os.makedirs("dist/data/class")
    except Exception as e:
        logger.warning(e)
    try:
        os.makedirs("dist/images/detail")
    except Exception as e:
        logger.warning(e)
    # 拷贝样式
    copy_tree("templates/assets", "dist/assets")
    copy_tree("templates/css", "dist/css")
    copy_tree("templates/js", "dist/js")
    copy_tree("templates/feedback", "dist/feedback")


def gen_page(pose):
    global age_range, df_record, class_names, pose_range
    if df_record is None:
        return None
    if pose=="00":
        page_name="index.html"
    else:
        page_name="face_" + pose + ".html"
    template = env.get_template("index.html")
    class_data = {}
    real_class = {}
    df_record1 = df_record[df_record['img_path'].str.find("/" + pose + "/")>=0]
    if df_record1 is not None and len(df_record1)>0:
        # 未按年龄分类的图片
        top10 = df_record1[df_record1['emotion'].isna()].sort_values('score', ascending=False).head(10).reset_index(drop=True)
        if len(top10)>0:
            top10['rank'] = top10.index + 1
            top10.loc[:, ['rank_class', 'rank_display']] = top10['rank'].apply(get_rank_class)
            class_data['undefined'] = {3: top10}
            real_class['undefined'] = class_names['undefined']
            # 拷贝图片到发布目录
            top10['img_path'].apply(copy_file)
            gen_details(top10)
        for k, v in class_names.items():
            data_k = {}
            total_k = 0
            for i in range(len(age_range)):
                top10 = df_record1[(df_record1['emotion']==k) & (df_record1['age']==i)].sort_values('score', ascending=False).head(10).reset_index(drop=True)
                if len(top10)>0:
                    top10['rank'] = top10.index + 1
                    top10.loc[:, ['rank_class', 'rank_display']] = top10['rank'].apply(get_rank_class)
                    data_k[i] = top10
                    # 拷贝图片到发布目录
                    top10['img_path'].apply(copy_file)
                    gen_details(top10)
                    logger.debug("add " + v + "_" + str(i))
                    total_k = total_k + len(top10)
            if total_k>0:
                real_class[k] = class_names[k]
                class_data[k] = data_k
    update_time = datetime.now().strftime("%Y-%m-%d")
    menus = class_data.keys()
    html_content = template.render(pose=pose, pose_range=pose_range, menus=menus, update_time=update_time, class_name=real_class, 
        class_data=class_data, age_range=age_range)
    with open("dist/" + page_name, 'w', encoding='utf8') as fout:
        fout.write(html_content)


def gen_one_detail(se):
    try:
        template = env.get_template("detail.html")
        img_id = se['img_id']
        page_name="detail/" + img_id + ".html"
        pose_imgs = glob.glob("dist/images/detail/" + img_id + "/*.jpg")
        pose_imgs = [img.replace("dist", "") for img in pose_imgs]
        html_content = template.render(img_id=img_id, pose_imgs=pose_imgs)
        with open("dist/" + page_name, 'w', encoding='utf8') as fout:
            fout.write(html_content)
    except Exception as e:
        logger.warning(e)
        return 0
    return 1


def gen_details(df):
    """
    生成详细介绍
    """
    result = df.apply(gen_one_detail, axis=1)
    logger.debug("df data " + str(len(df)) + " gen results " + str(sum(result)))


if __name__=="__main__":
    prepare_resource()
    for pose in pose_range.keys():
        gen_page(pose)
    logger.debug("gen index html finished")
