# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/17
import cv2
import numpy as np


def remove_the_bg(img):
    # 需灰度图
    img_canny = cv2.Canny(img, 127, 255)
    _, img_otsu = cv2.threshold(img_canny, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    binary = cv2.morphologyEx(img_otsu, cv2.MORPH_CROSS, np.ndarray((3, 3), np.uint8))
    # 掩膜
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.dilate(binary, se)
    res = cv2.add(cv2.bitwise_and(img, mask), cv2.bitwise_not(mask))
    return res


def k_means(X_data, k=3):
    X_data = np.array(X_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    return cv2.kmeans(X_data.astype(np.float32), k, None, criteria, 10, flags)