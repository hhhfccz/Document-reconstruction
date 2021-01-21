# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/8/25
import numpy as np
import cv2
import math


def rotated_img_with_fft(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 对图像进行边界扩充
    top_size, bottom_size, left_size, right_size = 50, 50, 50, 50
    img_new = cv2.copyMakeBorder(img_gray, top_size, bottom_size, left_size, right_size,
                                 borderType=cv2.BORDER_REPLICATE)

    # 获取图片大小，并针对DFT延拓
    h, w = img_new.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    img_new = cv2.copyMakeBorder(img_new, 0, bottom, 0, right,
                                 borderType=cv2.BORDER_CONSTANT, value=0)
    f = np.fft.fft2(img_new)
    f_shift = np.fft.fftshift(f)
    img_fft = np.log(np.abs(f_shift))

    # 取阈值，关键问题，阈值的判定对最终结果影响较大
    blur = cv2.GaussianBlur(img_fft, (3, 3), 1)
    thresh = cv2.Canny(blur.astype(np.uint8), 120, 200)

    # 霍夫变换，直线检测，关键问题
    lines = cv2.HoughLinesP(thresh.astype(np.uint8), 1.0, np.pi / 180, 40, minLineLength=40, maxLineGap=10)
    if lines is not None:
        lines1 = lines[:, 0, :]
    else:
        lines1 = []

    piThresh = np.pi / 180
    pi2 = np.pi / 2
    angle = 0
    for line in lines1:
        x1, y1, x2, y2 = line
        if abs(x2 - x1) < 1e-2:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
            if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
                continue
            else:
                angle = abs(theta)
                break

    angle = math.atan(angle)
    angle = angle / piThresh
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
    img = cv2.imread("E:\\cv\\Document-reconstruction\\rotated_result\\result1.jpg")
    ans = rotated_img_with_fft(img)
    cv2.imshow("img", ans)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
