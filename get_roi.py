# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/19
import cv2
import numpy as np


def get_roi(img_number, img, region):
    for i in range(len(region)):
        box = np.array(region[i])
        width = np.min(np.transpose(box), axis=1)
        height = np.max(np.transpose(box), axis=1)
        try:
            cv2.imwrite("./get_text_roi/img" + str(img_number) + "/" + str(i) + ".jpg",
                        img[width[1]:height[1], width[0]:height[0]])
        except cv2.error:
            print(i)
