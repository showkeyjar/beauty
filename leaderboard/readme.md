# 颜值排行榜

爬取人脸并评测颜值进行排名

目前包括：人脸表情排名

## todo 

1.人脸遮罩去除(目前未找到有效的模型)
    https://github.com/Oreobird/Face-Occlusion-Detect

2.人脸部位排名

3.理想人脸搜索

4.参与排行


## 流程

环境准备 python 3.8 (注意部分包不支持更高版本)
conda install pyarrow

1.抓取图片

    cd leaderboard/AutoCrawler
    python main.py --skip true --threads 8 --google true --naver true --full false --face true --no_gui true --limit 0
    (第二次抓取时会提示imbalance的warning,可以忽略)

2.提取人脸

    python -m leaderboard.ex_face

3.人脸去重

    python -m leaderboard.drop_duplicates

4.人脸重建(3d姿态估计与校准)

    cd leaderboard/Face-Pose-Net
    conda activate tf1
    python gen_pose.py

    尝试使用 https://github.com/jiaxiangshang/MGCNet 改进人脸重建结果

5.再次去重(由于生成人脸后分布又不同了，可选)

    cd leaderboard/
    python unique_face.py
    
6.人脸分类(old)

    cd leaderboard/Facial-Expression-Recognition.Pytorch
    python auto_classify.py

6.人脸分类(New)

    cd leaderboard/
    python face_class.py --f 1

7.人脸评分

    cd leaderboard/
    python auto_rank.py

8.爬取候选人信息

    参考:https://www.makeuseof.com/tag/3-fascinating-search-engines-search-faces/

9.发布排行(仅包含亚洲女性)

    cd leaderboard/
    python gen_html.py
