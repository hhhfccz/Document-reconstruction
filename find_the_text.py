# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/14
import cv2
import numpy as np
from get_roi import get_roi


def preprocess(img_gray):
    # 二值化
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_OTSU)
    # 开运算
    return cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=2)


def find_text_region(img):
    # 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(img)
    region = []
    # 查找轮廓
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print(r"rect is: {}".format(str(rect)))
        # box是四个点的坐标
        box = np.int0(cv2.boxPoints(rect))
        # 筛选
        width, height = abs(box[0] - box[2])
        if height > 1.2 * width:
            continue
        region.append(box)
    return region


def detect(img_number, img):
    region = find_text_region(img)
    # get_roi(img_number, img, region)
    print(len(region))
    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
    return img


if __name__ == '__main__':
    pass
