# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/14
import cv2
import numpy as np
from get_roi import get_roi


def detect(img_gray, number):
    # 创建mser实例
    mser = cv2.MSER_create(_delta=2, _min_area=200, _max_variation=0.7)
    # 检测
    regions, boxes = mser.detectRegions(img_gray)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img_gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return 


if __name__ == '__main__':
    pass
