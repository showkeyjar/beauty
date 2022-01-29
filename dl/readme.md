# DeepLearning

## 推理

采用deeplearning方法对人脸打分

### 1.使用 autokeras 生成模型结构；

![ak net](../img/s1.png)

### 2.使用 tensorflow 训练权重；

启动tensorboard:

    tensorboard --logdir=../logs

### 3.加载模型预测；


## 训练

### 1.使用 fast-autoaugment 对图像增强；

### 2.使用 Auto-PyTorch 训练模型；

    https://github.com/automl/Auto-PyTorch


# BarlowTwins

使用两部实现：

1.在视频流中区分出最优正面(人脸识别)；

2.使用现有评分等级归类到最优分(人脸评分)；

## 1.人脸识别

### 训练方法

https://github.com/facebookresearch/barlowtwins

nohup python main.py data/SCUT-FBP5500_v2/Images/ --batch-size 256 > train_face.log &

mv checkpoint/resnet50.pth model/barlowtwins/face.pth

## 2.人脸评分(old)

nohup python main.py data/SCUT-FBP5500_v2/score_data/ --batch-size 256 > train_score.log &

mv checkpoint/resnet50.pth model/barlowtwins/score.pth

实际测试结果不理想

## 2.人脸评分(new)

参考：keras-efficientnet-regression/readme.md

## 3.皮肤识别

1. 准备数据:
    cd dl/scan/0.get_skin_dlib.py
    conda activate dlib
    python 0.get_skin_dlib.py

2. 安装solo-learn
    pip install torch torchvision torchaudio


## 常见问题：

1. ImportError: libjpeg.so.9: cannot open shared object file: No such file or directory
    通常都是由于环境没能完全同步导致的，将lib下的文件拷贝过去即可

2. ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by /opt/anaconda3/
    cp /opt/anaconda3/lib/libstdc++.so.6.0.26 /lib64/
    rm -rf /lib64/libstdc++.so.6
    ln -s /lib64/libstdc++.so.6.0.26 /lib64/libstdc++.so.6

# todo

1.人脸皮肤训练评估 (使用常规CNN学习分类)

2.人脸3d重建，角度分析(预估人脸的角度)

3. https://github.com/zllrunning/face-parsing.PyTorch 对分类混乱的问题做了答复，可以验证下

4. 如果IIC及scan都不理想，可以尝试使用 solo learn 尝试聚类 https://github.com/vturrisi/solo-learn

5. 颜值报告可以尝试使用 XAI 解释(主要用于机器学习，可能不恰当)
