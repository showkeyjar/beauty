# %% coding=utf-8
import os
import dlib         # 机器学习的库 Dlib
import cv2          # 图像处理的库 OpenCv
import time
import timeit
import requests
import statistics
import webbrowser
import pandas as pd
from sklearn.externals import joblib

"""
调用摄像头，进行人脸捕获，和 68 个特征点的追踪
Author:   coneypo
Blog:     http://www.cnblogs.com/AdaminXie
GitHub:   https://github.com/coneypo/Dlib_face_detection_from_camera
Created at 2018-02-26
Updated at 2019-01-28

todo 需要添加人脸对齐
"""

# 储存截图的目录
path_screenshots = "img/screenshots/"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')
model = joblib.load('model/beauty.pkl')

# 创建 cv2 摄像头对象
cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数，propId 设置的视频参数，value 设置的参数值
cap.set(3, 480)

# 截图 screenshots 的计数器
cnt = 0
best_score = 0
best_img = None
time_cost_list = []


def prepare_input(img, face):
    f_width = abs(face.right() - face.left())
    f_height = abs(face.bottom() - face.top())
    shape = predictor(img, face)
    # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
    face_shape = {}
    for i in range(0, 67):
        for j in range(i + 1, 68):
            face_shape[str(i) + '_' + str(j) + '_x'] = abs(shape.part(i).x - shape.part(j).x) / f_width
            face_shape[str(i) + '_' + str(j) + '_y'] = abs(shape.part(i).y - shape.part(j).y) / f_height
            # print(str(i) + '_' + str(j))
    # shape_size.append(face_shape)
    df_image = pd.DataFrame.from_dict([face_shape])
    return df_image


def gen_upload(img_path):
    cap.release()
    with open(img_path, 'rb') as f1:
        files = [
            ('sc', f1)
        ]
        # flask 服务地址
        resp = requests.post('http://172.16.254.164:5000/upload_gen', files=files)
        path = os.path.abspath('data/reports/temp.htm')
        url = 'file://' + path
        with open(path, 'w') as f:
            f.write(str(resp.content))
        webbrowser.open(url)


# cap.isOpened() 返回 true/false 检查初始化是否成功
while cap.isOpened():
    # cap.read()
    # 返回两个值：
    #    一个布尔值 true/false，用来判断读取视频是否成功/是否到视频末尾
    #    图像对象，图像的三维矩阵
    flag, im_rd = cap.read()
    # 每帧数据延时 1ms，延时为 0 读取的是静态帧
    k = cv2.waitKey(1)
    # 取灰度
    img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
    # start point
    start = timeit.default_timer()
    # 人脸数
    face = detector(img_gray, 1)
    # print(len(faces))
    # 待会要写的字体
    font = cv2.FONT_HERSHEY_SIMPLEX
    # 标 68 个点
    pred = 0
    if len(face) != 0:
        # 检测到人脸
        #for i in range(len(faces)):
            #landmarks = np.matrix([[p.x, p.y] for p in predictor(im_rd, faces[i]).parts()])
            #for p in predictor(im_rd, faces[i]).parts():
            # for idx, point in enumerate(landmarks):
            #     # 68 点的坐标
            #     pos = (point[0, 0], point[0, 1])
            #
            #     # 利用 cv2.circle 给每个特征点画一个圈，共 68 个
            #     cv2.circle(im_rd, pos, 2, color=(139, 0, 0))
            #
            #     # 利用 cv2.putText 输出 1-68
            #     cv2.putText(im_rd, str(idx + 1), pos, font, 0.2, (187, 255, 255), 1, cv2.LINE_AA)
        for i, d in enumerate(face):
            df_image = prepare_input(im_rd, d)
            pred = model.predict(df_image)
            if pred > best_score:
                best_score = pred
                if best_img is not None:
                    os.remove(best_img)
                best_img = path_screenshots + "screenshot" + "_" + str(cnt) + "_" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg"
                cv2.imwrite(best_img, im_rd)
            #cv2.putText(im_rd, "face score: " + str(pred), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        #cv2.putText(im_rd, "faces: " + str(len(faces)), (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        # end point
        stop = timeit.default_timer()
        time_cost_list.append(stop - start)
        #print("%-15s %f" % ("Time cost:", (stop - start)))
    else:
        # 没有检测到人脸
        cv2.putText(im_rd, "no face", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
    # 添加说明
    im_rd = cv2.putText(im_rd, "best score : " + str(best_score), (20, 350), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    im_rd = cv2.putText(im_rd, "press 'G': gen report", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    im_rd = cv2.putText(im_rd, "press 'Q': quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    # 按下 'g' 键生成报告
    if k == ord('g'):
        cnt += 1
        print(best_img)
        im_rd = cv2.imread(best_img)
        im_rd = cv2.putText(im_rd, "generating report...", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        gen_upload(best_img)
    # 按下 'q' 键退出
    if k == ord('q'):
        break

    # 窗口显示
    # 参数取 0 可以拖动缩放窗口，为 1 不可以
    # cv2.namedWindow("camera", 0)
    cv2.namedWindow("camera", 1)
    cv2.imshow("camera", im_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()

print("%-15s" % "Result:")
print("%-15s %f" % ("Max time:", (max(time_cost_list))))
print("%-15s %f" % ("Min time:", (min(time_cost_list))))
print("%-15s %f" % ("Average time:", statistics.mean(time_cost_list)))
