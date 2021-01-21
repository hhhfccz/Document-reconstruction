# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/8/25
import numpy as np
import cv2
import math


def get_angle(lines):
    number_inexistence_k = 0
    sum_positive_k45 = 0
    number_positive_k45 = 0
    sum_positive_k90 = 0
    number_positive_k90 = 0
    sum_negative_k45 = 0
    number_negative_k45 = 0
    sum_negative_k90 = 0
    number_negative_k90 = 0
    number_zero_k = 0
    for x in lines:
        if x[2] == x[0]:
            number_inexistence_k += 1
            continue
        if 0 < math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0]))) < 45:
            number_positive_k45 += 1
            sum_positive_k45 += math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0])))
        if 45 <= math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0]))) < 90:
            number_positive_k90 += 1
            sum_positive_k90 += math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0])))
        if -45 < math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0]))) < 0:
            number_negative_k45 += 1
            sum_negative_k45 += math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0])))
        if -90 < math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0]))) <= -45:
            number_negative_k90 += 1
            sum_negative_k90 += math.degrees(math.atan((x[3] - x[1]) / (x[2] - x[0])))
        if x[3] == x[1]:
            number_zero_k += 1
    max_number = max(number_inexistence_k, number_positive_k45, number_positive_k90, number_negative_k45,
                     number_negative_k90, number_zero_k)
    if max_number == number_inexistence_k:
        return 90
    if max_number == number_positive_k45:
        return sum_positive_k45 / number_positive_k45
    if max_number == number_positive_k90:
        return sum_positive_k90 / number_positive_k90
    if max_number == number_negative_k45:
        return sum_negative_k45 / number_negative_k45
    if max_number == number_negative_k90:
        return sum_negative_k90 / number_negative_k90
    if max_number == number_zero_k:
        return 0


def rotated_img_with_fft(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对图像进行边界扩充
    top_size, bottom_size, left_size, right_size = 50, 50, 50, 50
    img_new = cv2.copyMakeBorder(img_gray, top_size, bottom_size, left_size, right_size,
                                 borderType=cv2.BORDER_REPLICATE)
    h, w = img_new.shape[:2]

    # 取阈值，关键问题，阈值的判定对最终结果影响较大
    blur = cv2.GaussianBlur(img_new, (3, 3), 1)
    thresh = cv2.Canny(blur.astype(np.uint8), 127, 255)

    # 霍夫变换，直线检测，关键问题
    lines = cv2.HoughLinesP(thresh.astype(np.uint8), 1.0, np.pi/180, 100, minLineLength=100, maxLineGap=20)
    if lines is not None:
        lines = lines[:, 0, :]
    else:
        lines = []

    angle = get_angle(lines)
    center = (w // 2, h // 2)
    height_1 = int(w * math.fabs(math.sin(math.radians(angle))) + h * math.fabs(math.cos(math.radians(angle))))
    width_1 = int(h * math.fabs(math.sin(math.radians(angle))) + w * math.fabs(math.cos(math.radians(angle))))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (width_1 - w) / 2
    M[1, 2] += (height_1 - h) / 2

    # 旋转
    rotated = cv2.warpAffine(img_gray, M, (width_1, height_1), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    rotated = cv2.resize(rotated, (int(width_1), int(height_1)), interpolation=cv2.INTER_CUBIC)

    return rotated


if __name__ == "__main__":
    pass
