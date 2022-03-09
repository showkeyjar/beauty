# 移动解决方案

## 功能设计

1. 评测

    颜值评测
    年龄评测
    缺陷生成
    解决方案
    
    删除并重新设计颜值分析
    
    通过使用MLKit自带的 133 个人脸关键点提取皮肤特征
    
    https://developers.google.com/ml-kit/vision/face-detection/android

    ![face point](img/face_contours.svg)
    

2. 报表

    颜值变化趋势
    年龄变化趋势
    缺陷变化趋势
    方案执行情况
    解决效果评估

3. 排行

    表情\角度\年龄排行
    参与排行
    颜值PK
    错误反馈

## todo

1.友盟+集成隐私策略实现逻辑

2.参与排行功能

3.社会化分享

4.动态下载模型：
    参考:https://github.com/AManCallJiang/AwesomeDownloader-Android

## 使用androidstudio 开发

解决Room访问不到数据的问题:
https://stackoverflow.com/questions/69313549/myroomdatabase-impl-does-not-exist/69313951#69313951


## Kivy 生成移动应用

pip install python-for-android


p4a apk --private $HOME/code/myapp --package=org.example.myapp --name "My application" --version 0.1 --bootstrap=sdl2 --requirements=python3,kivy

## BeeWare 生成项目


    pip install briefcase

    # 可选(新安装环境时，需执行该步骤创建一个test项目可自动安装GUI组件)
    cd App
    briefcase new
    cd test
    briefcase create
    briefcase build
    # briefcase run
    # briefcase package
    # briefcase update
    briefcase create android
    # 这里会自动下载android sdk，如果安装错误，可以手工删除重新下载
    # SDK地址：C:\Users\[用户名]\.briefcase\tools\android_sdk
    briefcase build android
    briefcase run android
    # 这里会自动创建 emulator，如果创建模拟器错误，可以手工删除重新创建
    # 模拟器地址：C:\Users\[用户名]\.android\avd
    
    cd beauty
    
    briefcase dev

## 问题：

1. briefcase emulator错误警告：
    https://developer.android.com/studio/run/emulator-acceleration#disable-hyper-v
    按照官方文档禁用 hyper-v 还是不够的，还需要在windows服务中将 hyper-v 相关服务关闭
    重启登录cmos，打开VT-x功能（如果有条件）
    在已经支持 WHPX 的机器上，模拟器总是无法启动，暂不清楚原因

2. briefcase 原生使用 toga, toga目前暂不支持调用android摄像头

Train:

install:

conda install -c moussi gcc_impl_linux-64
# ln -s /share2/home/anconda3/envs/my_env/libexec/gcc/x86_64-conda_cos6-linux-gnu/7.3.0/gcc /share2/home/anaconda3/my_env/bin/gcc
ln -s /opt/anaconda3/envs/pytorch/libexec/gcc/x86_64-conda-linux-gnu/9.3.0/gcc /opt/anaconda3/envs/pytorch/bin/gcc

ln -s /opt/anaconda3/envs/pytorch/libexec/gcc/x86_64-conda-linux-gnu/9.3.0/g++-4.9 /opt/anaconda3/envs/pytorch/bin/g++

conda install gcc_linux-64
conda install gxx_linux-64
conda deactivate
conda activate my_en
gcc -v

ln -s /opt/anaconda3/envs/pytorch/bin/x86_64-conda-linux-gnu-gcc /opt/anaconda3/envs/pytorch/bin/gcc
ln -s /opt/anaconda3/envs/pytorch/bin/x86_64-conda-linux-gnu-g++ /opt/anaconda3/envs/pytorch/bin/g++

3. 快速切换到“排行”时，应用会崩溃


### yolox训练

conda create -p /opt/conda-env/yolox python=3.8
conda activate /opt/conda-env/yolox

以上9.3.0版本的gcc无法成功编译 yolox


git clone https://github.com/NVIDIA/apex.git
python setup.py install

##### 注意这里只能用gcc-5
conda install -c psi4 gcc-5

##### 安装apex前需要安装完整的pytorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia

pip3 install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

conda install -c nvidia nvcc_linux-64

##### 使用conda提供的cudatoolkit不包含nvcc，需要安装完整的cuda以及cudnn

yolox best AP 7.8 各分类的准确率太低
