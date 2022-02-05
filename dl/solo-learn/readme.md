# 使用solo-learn训练模型

some one comment sololearn is easy than vissl:
https://www.reddit.com/r/MachineLearning/comments/oka0v7/p_sololearn_a_library_of_selfsupervised_methods/

check solo-learn vs vissl vs lightly

https://github.com/lightly-ai/lightly

git clone https://github.com/vturrisi/solo-learn.git

由于solo-learn byol训练输出的结果是一堆图像特征，所以还需要在最后增加一层分类器

选择lda分类器: https://github.com/rmsander/spatial_LDA

## 0.取得局部数据

    0.get_face_part_dlib.py

## 1.皮肤模型

    skin/
    train_skin.sh
    save_byol.py
    predict.py
    train_lda.ipynb

## 2.额头模型

    forehead/
    train_forehead.sh

