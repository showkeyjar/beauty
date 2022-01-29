import cv2
import dlib
import math
import numpy as np
from os.path import dirname, join
"""
使用dlib获取不同位置
部位评分 无需使用dlib，google ml-kit face-detection模型也支持人脸关键点及轮廓点的提取
dlib用于训练部位分类模型
https://developers.google.com/ml-kit/vision/face-detection/android

注意部位都是彩图，不是二维 [x,y,ch],所以不能轻易 reshape

create contour
https://stackoverflow.com/questions/14161331/creating-your-own-contour-in-opencv-using-python

todo contours 以顺时针顺序连接，注意检查
"""
dlib_face_part = {
    "center":
    {
        "left_eyebrow": [17, 18, 19, 20, 21],
        "right_eyebrow": [22, 23, 24, 25, 26],
        "left_upper_eyelid": (36, 37, 38, 39),
        "right_upper_eyelid": (42, 43, 44, 45),
        "left_lower_eyelid": (36, 41, 40, 39),
        "right_lower_eyelid": (42, 47, 46, 45),
        "nose_bridge": [27, 28, 29, 30],
    },
    "inside":
    {
        "left_eye": [36, 37, 38, 39, 40, 41],
        "right_eye": [42, 43, 44, 45, 46, 47],
        "nose_tip": [30, 31, 32, 33, 34, 35],
        "mouse": [60, 61, 62, 63, 64, 65, 66, 67],
        "upper_lip": [48, 49, 50, 51, 52, 53, 54, 60, 61, 62, 63, 64],
        "lower_lip": [60, 67, 66, 65, 64, 48, 59, 58, 57, 56, 55, 54],
        "jaw": [48, 59, 58, 57, 56, 55, 54, 5, 6, 7, 8, 9, 10, 11],
        "left_cheek": [1, 36, 41, 40, 39, 31, 48, 3, 2],
        "right_cheek": [42, 47, 46, 45, 15, 14, 13, 54, 35],
    },
    "top":
    {
        "forehead": [0, 17, 18, 19, 20, 21, 27, 22, 23, 24, 25, 26, 16],
    }
}


detector = dlib.get_frontal_face_detector()
model_file = join(dirname(__file__), "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(model_file)


def array_to_img(full_data):
    full_data = full_data[:len(full_data) - len(full_data)%3]
    part_value = full_data.reshape(-1, 3)
    part_value_len = part_value.shape[0]
    # 生成图像数组(图片)方便卷积
    part_width = math.floor(math.sqrt(part_value_len))
    part_value = part_value[:part_width**2,:]
    part_value = part_value.reshape(part_width, part_width, 3)
    return part_value


def get_center_part_value(img, bones, round_range=3):
    """
    todo 根据中间骨骼取得外部数据
    """
    part_mask = []
    b0 = bones[0]
    for i in range(1, len(bones)):
        b1 = bones[i]
        x0, y0 = b0[1], b0[0]
        x1, y1 = b1[1], b1[0]
        length = int(np.hypot(x1-x0, y1-y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        z_index = np.indices(img.shape)
        for p_y, p_x in zip(y.astype(np.int), x.astype(np.int)):
            # zmask = np.argwhere((z_index[0] > p_y-round_range) & (z_index[1] > p_x-round_range) & (z_index[0] < p_y+round_range) & (z_index[1] < p_x+round_range))
            zmask = np.ma.masked_where((z_index[0] > p_y-round_range) & (z_index[1] > p_x-round_range) & (z_index[0] < p_y+round_range) & (z_index[1] < p_x+round_range), img)
            # zmask = np.logical_and.reduce((img[:,:,0]>100,img[:,:,1]>100,img[:,:,2]>100))
            part_mask.append(zmask)
        b0 = bones[i]
    full_mask = np.logical_or.reduce(part_mask)
    full_data = img[full_mask].reshape(-1)
    part_value = array_to_img(full_data)
    return part_value


def get_out_contours(bones, type="inside", point_range=3):
    """
    todo 根据骨骼取得轮廓
    查询 cv2 dilate

    找出最大轮廓,参考: https://blog.csdn.net/Janine_1991/article/details/114675147
    contours,hierarchy=cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)
    mask=cv2.drawContours(img,contours,0,255,cv2.FILLED)

    """
    contours = []
    # 假设 bones 是按顺时针排列的
    for b in bones:
        c = [b[0] - point_range, b[1] - point_range]
        contours.append(c)
    
    return contours


def get_inside_part_value(img, bones):
    """
    根据骨骼取得内部数据
    参考: https://www.codetd.com/en/article/12758906
    """
    grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(grayimg)
    # 填充任意形状
    # cv2.fillPoly(mask, [np.array(bones)], (255), 8, 0)
    # 填充凸多边形(实测与fillPoly效果一样)
    cv2.fillConvexPoly(mask, np.array(bones), (255), 8, 0)
    result = cv2.bitwise_and(img, img, mask=mask)
    cnts = sorted([np.array(bones)], key=cv2.contourArea, reverse=True)
    # Find bounding box and extract ROI
    x,y,w,h = cv2.boundingRect(cnts[0])
    ROI = result[y:y+h, x:x+w]
    return ROI


def get_top_part_value(img, bones, round_range=12):
    """
    todo 根据底部骨骼取得上部数据
    """
    part_mask = []
    b0 = bones[0]
    for i in range(1, len(bones)):
        b1 = bones[i]
        x0, y0 = b0[1], b0[0]
        x1, y1 = b1[1], b1[0]
        length = int(np.hypot(x1-x0, y1-y0))
        x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
        z_index = np.indices(img.shape)
        for p_y, p_x in zip(y.astype(np.int), x.astype(np.int)):
            zmask = np.ma.masked_where((z_index[0] > p_y) & (z_index[1] == p_x) & (z_index[0] < p_y+round_range), img)
            part_mask.append(zmask)
        b0 = bones[i]
    full_mask = np.logical_or.reduce(part_mask)
    full_data = img[full_mask].reshape(-1)
    part_value = array_to_img(full_data)
    return part_value


def get_part_value(img, bones, ptype="center", round_range=3):
    """
    取得不同部位的得分
    """
    if ptype=="center":
        return get_center_part_value(img, bones, round_range=round_range)
    elif ptype=="inside":
        return get_inside_part_value(img, bones)
    else:
        return get_top_part_value(img, bones, round_range=round_range)


def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords


def get_face_parts(img, ptype=None):
    """
    根据评测结果图片返回各部分得分
    """
    global dlib_face_part
    part_result = {}
    dets = detector(img, 1)
    if len(dets)>0:
        det = dets[0]
        shape = predictor(img, det)
        if ptype is None:
            for p_type,p in dlib_face_part.items():
                # p_type = center left
                for p_name, b_indexs in p.items():
                    # p_name = left_eyebrow
                    bones = [[shape.part(i).x, shape.part(i).y] for i in b_indexs]
                    part_result[p_name] = get_part_value(img, bones, p_type)
        else:
            p = dlib_face_part[ptype]
            for p_name, b_indexs in p.items():
                # p_name = left_eyebrow
                bones = [[shape.part(i).x, shape.part(i).y] for i in b_indexs]
                part_result[p_name] = get_part_value(img, bones, ptype)
    return part_result


def get_contour_value(img, bones):
    return get_inside_part_value(img, bones)


def test(img_path):
    img = cv2.imread(img_path)
    ret = get_face_parts(img)
    print(ret)


if __name__=="__main__":
    test("test.jpg")
