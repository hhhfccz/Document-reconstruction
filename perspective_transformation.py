# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/12/1
import cv2
import numpy as np
filename = "E:\\cv\\Document-reconstruction\\rotated_result\\result2.jpg"
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
# 输入图像必须是float32，最后一个参数在0.04 到0.06 之间
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01*dst.max()] = [0, 0, 255]
cv2.imshow('dst', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Shi-Tomasi角点检测
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)

for i in np.int0(corners):
    # 压缩至一维：[[62, 64]] -> [62, 64]
    x, y = i.ravel()
    cv2.circle(img, (x, y), 4, (0, 0, 255), -1)

# cv2.imwrite('Shi-Tomasi-corner.jpg', img)
cv2.imshow('dst', img)
cv2.waitKey(0)