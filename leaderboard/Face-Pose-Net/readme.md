# 人脸姿态预测

使用 tensorflow 1.0 运行

conda activate tf1
conda install -c conda-forge opencv
conda install -c conda-forge python-lmdb
conda install scikit-learn matplotlib
pip install pyarrow

姿态预测

python main_predict_6DoF.py 0 input_test.txt

姿态渲染

python main_fpn.py input.csv
