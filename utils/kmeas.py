# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2021/5/12
import numpy as np
import cv2


def k_means(X_data, k=3):
    X_data = np.array(X_data)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    return cv2.kmeans(X_data.astype(np.float32), k, None, criteria, 10, flags)
