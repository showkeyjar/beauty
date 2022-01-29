# 无监督学习识别皮肤类型

参考：https://github.com/wvangansbeke/Unsupervised-Classification

https://github.com/hkhailee/GOLDEN_unsupervised/blob/main/TrainingYourOwnDataset.md

发现另外一款无监督框架：https://github.com/xu-ji/IIC

由于 scan.py框架的数据集加载部分的编码，大量混用了标签数据，单独抽取出来修改比较麻烦，改尝试IIC


## 安装

    conda install -c conda-forge faiss-gpu

    conda install -c conda-forge pyyaml easydict termcolor

## 执行

1. 创建 configs/scan/scan_skin.yml 创建先验知识

2. 创建数据集 data/skin.py 

3. 执行 python scan.py --config_env configs/env.yml --config_exp configs/scan/scan_skin.yml

执行过程过于复杂，尝试使用IIC (IIC训练缓慢，需要检查训练结果)

