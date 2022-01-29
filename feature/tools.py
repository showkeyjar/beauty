import os
import cv2
import dill
import dlib
import math
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import local_binary_pattern

"""
特征工具包
"""

predictor_path = "model/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

with open('data/result_cols.txt', 'r') as f:
    select_str = str(f.readline())
select_cols = select_str.split(' ')

# 用于存储人脸68个点位的坐标
face_points = {}
load_img = None
load_path = None


def get_image_hull_mask(image_shape, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)

    #hull_mask = np.zeros(image_shape[0:2]+(1,), dtype=np.float32)
    hull_mask = np.full(image_shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[17:18]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))

    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    return hull_mask


# 加入alpha通道 控制透明度
def merge_add_alpha(img_1, mask):
    # merge rgb and mask into a rgba image
    r_channel, g_channel, b_channel = cv2.split(img_1)
    if mask is not None:
        alpha_channel = np.ones(mask.shape, dtype=img_1.dtype)
        alpha_channel *= mask*255
    else:
        alpha_channel = np.zeros(img_1.shape[:2], dtype=img_1.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA


def merge_add_mask(img_1, mask):
    if mask is not None:
        height = mask.shape[0]
        width = mask.shape[1]
        channel_num = mask.shape[2]
        for row in range(height):
            for col in range(width):
                for c in range(channel_num):
                    if mask[row, col, c] == 0:
                        mask[row, col, c] = 0
                    else:
                        mask[row, col, c] = 255

        r_channel, g_channel, b_channel = cv2.split(img_1)
        r_channel = cv2.bitwise_and(r_channel, mask)
        g_channel = cv2.bitwise_and(g_channel, mask)
        b_channel = cv2.bitwise_and(b_channel, mask)
        res_img = cv2.merge((b_channel, g_channel, r_channel))
    else:
        res_img = img_1
    return res_img


def get_landmarks(image):
    global predictor_path
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(img_gray, 0)
    landmarks = None
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image, rects[i]).parts()])
    return landmarks


def cut_face_img(image, img_size=(186, 186)):
    """
    裁剪人脸部分的图像
    :param image:
    :param landmarks:
    :return: 186 * 186 image
    """
    if image is None:
        return None
    dets = detector(image, 1)
    format_img = None
    if len(dets) > 0:
        try:
            d = dets[0]
            cropped = image[d.top():d.bottom(), d.left():d.right()]
            if cropped is not None:
                format_img = cv2.resize(cropped, img_size)
        except Exception as e:
            print(e)
    return format_img


def get_seg_face(img_path):
    try:
        image = cv2.imread(img_path)
    except Exception as e:
        print(e)
        print(img_path)
        image = None
    image_bgr = None
    landmarks = None
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        landmarks = get_landmarks(image)
        if landmarks is not None:
            mask = get_image_hull_mask(np.shape(image), landmarks).astype(np.uint8)
            # image_bgra = merge_add_alpha(image, mask)
            # cv2.imwrite("static/uploads/tmp/Messi_add_alpha.jpg", image_bgra)
            image_bgr = merge_add_mask(image, mask)
            # cv2.imwrite("static/uploads/tmp/Messi_add_mask.jpg", image_bgr)
    return image_bgr, landmarks


def single_face_alignment(face, landmarks):
    if landmarks is None:
        return None
    eye_center = ((landmarks[36, 0] + landmarks[45, 0]) * 1. / 2,  # 计算两眼的中心坐标
                  (landmarks[36, 1] + landmarks[45, 1]) * 1. / 2)
    dx = (landmarks[45, 0] - landmarks[36, 0])  # note: right - right
    dy = (landmarks[45, 1] - landmarks[36, 1])
    angle = math.atan2(dy, dx) * 180. / math.pi  # 计算角度
    RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
    align_face = cv2.warpAffine(face, RotateMatrix, (face.shape[0], face.shape[1]))  # 进行放射变换，即旋转
    return align_face


def face_correct(img_path, gen_file=True, new_path=None):
    """
    人脸矫正
    包括：对齐、旋转、缩放
    :param img_path:
    :param gen_file:
    :param new_path:
    :return:
    """
    result = None
    image_bgra, landmarks = get_seg_face(img_path)
    if image_bgra is not None:
        rot_img = single_face_alignment(image_bgra, landmarks)
        # 使用cv2操作图片之前，不要做bgr2rgb的转换
        # rgb_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)
        image_bgr = cut_face_img(rot_img)
        if gen_file and image_bgr is not None:
            if new_path is None:
                new_path = img_path.replace('.jpg', '') + '_face.jpg'
            cv2.imwrite(new_path, image_bgr)
            result = new_path
        else:
            result = image_bgr
    return result


def lbph_extract(img, d=None, eps=1e-7):
    """
    生成lbph特征
    :param img:
    :param d:
    :param eps:
    :return:
    """
    no_points = 24
    radius = 8
    d_top = d.top()
    d_bottom = d.bottom()
    d_left = d.left()
    d_right = d.right()
    cropped = img[d_top:d_bottom, d_left:d_right]
    im_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
    (histogram, _) = np.histogram(lbp.ravel(),
                                  bins=np.arange(0, no_points + 3),
                                  range=(0, no_points + 2))
    # now we need to normalise the histogram so that the total sum=1
    histogram = histogram.astype("float")
    histogram /= (histogram.sum() + eps)
    skin_hist = {}
    for i, h in enumerate(histogram):
        skin_hist['skin_' + str(i)] = h
    return skin_hist


# %%
def prepare_input(img_path):
    """
    预测准备(生成特征)
    :param img_path:
    :return:
    """
    global select_cols, face_points, load_img, load_path
    img = dlib.load_rgb_image(img_path)
    load_path = img_path
    # 上传时即矫正图片，这里不需要再矫正
    # head_img = face_correct(img_path)
    load_img = img
    dets = detector(load_img, 1)
    d = dets[0]
    # for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
    f_width = abs(d.right() - d.left())
    f_height = abs(d.bottom() - d.top())
    # print('width:' + str(f_width) + ', height:' + str(f_height))
    # Get the landmarks/parts for the face in box d.
    shape = predictor(load_img, d)
    # print("Part 0: {}, Part 1: {} ...".format(shape.part(0), shape.part(1)))
    face_shape = {}
    for i in range(0, 67):
        for j in range(i + 1, 68):
            face_shape[str(i) + '_' + str(j) + '_x'] = abs(shape.part(i).x - shape.part(j).x) / f_width
            face_shape[str(i) + '_' + str(j) + '_y'] = abs(shape.part(i).y - shape.part(j).y) / f_height
            p_left = max(min(shape.part(i).x, shape.part(j).x) - 5, 0)
            p_upper = max(min(shape.part(i).y, shape.part(j).y) - 5, 0)
            p_right = min(max(shape.part(i).x, shape.part(j).x) + 5, 186)
            p_lower = min(max(shape.part(i).y, shape.part(j).y) + 5, 186)
            face_points[str(i) + '_' + str(j)] = (p_left, p_upper, p_right, p_lower)
            # print(str(i) + '_' + str(j))
    # shape_size.append(face_shape)
    skin_hists = lbph_extract(img, d)
    face_shape.update(skin_hists)
    face_shape = {k: v for k, v in face_shape.items() if k in select_cols}
    df_image = pd.DataFrame.from_dict([face_shape])
        # break
    # df_image = {k: v for k, v in df_image.items() if k in select_cols}
    # print(df_image)
    return df_image


def get_feature_points(feature_name):
    """
    取得特征坐标
    :param feature_name:
    :return:
    """
    global face_points
    try:
        points = face_points[feature_name]
    except:
        points = None
    return points


def gen_feature_pic(feature_name):
    global load_path
    img = Image.open(load_path)
    try:
        save_path = load_path.split('.')[0] + "/"
    except:
        save_path = 'static/uploads/tmp/'
    try:
        os.makedirs(save_path)
    except:
        pass
    points = get_feature_points(feature_name)
    if points is not None:
        # cropped = load_img[0:128, 0:512]
        # cv2.imwrite("./data/cut/cv_cut_thor.jpg", cropped)
        cropped = img.crop(points)
        cropped.save(save_path + feature_name + ".jpg")


def get_feature_value(x):
    global data_path
    df_image = prepare_input(data_path + 'Images/' + x)
    return df_image


def gen_best_feature():
    """
    生成美人特征
    :return:
    """
    data_path = 'G:/data/SCUT-FBP5500_v2/'
    data = pd.read_excel(data_path + 'All_Ratings.xlsx', None)
    df_asian = data['Asian_Female']
    df_data = df_asian.groupby('Filename').agg(np.mean)
    df_data['file'] = df_data.index.astype(str)

    best_data = df_data[df_data['Rating'] >= 4]
    best_value = best_data['file'].apply(get_feature_value)
    df_best = pd.concat(best_value.to_list())
    se_best = df_best.apply(np.mean, axis=0)
    with open('model/best_values.pkl', 'wb') as f:
        dill.dump(se_best, f)


if __name__ == "__main__":
    # get_seg_face('static/uploads/test1.jpg')
    img = face_correct('static/uploads/test1.jpg')
    cv2.imwrite("static/uploads/tmp/test1_face.jpg", img)
