# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/8/25
import numpy as np
import cv2
from remove_the_background import remove_the_bg


def get_match_img(img_left, img_right, number, MIN_MATCH_COUNT=10, norm=0.75):
    # 获取图片大小，调整图像大小，使得两张图像大小相同
    h, w = img_left.shape[:2]

    # 取出底色，避免干扰
    img_left_g = remove_the_bg(cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY))
    img_right_g = remove_the_bg(cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY))

    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 寻找关键点和描述符
    keypoint1, features1 = sift.detectAndCompute(img_left_g, None)
    keypoint2, features2 = sift.detectAndCompute(img_right_g, None)

    # 设置FLANN参数，如果取0极为耗费时间
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)

    # 将FlannBasedMatcher方法实例化
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 利用knnMatch匹配处理，并将结果返回给matches，请确保k=2
    matches = flann.knnMatch(features1, features2, k=2)

    if len(matches) > MIN_MATCH_COUNT:
        # 得到两幅待拼接图的匹配点集
        good = []
        for m, n in matches:
            if m.distance < norm * n.distance:
                good.append(m)
        src_pts = np.float32([keypoint1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoint2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # cv2.findHomography：传入两个图像里的点集合,返回目标的透视转换
        # 第三个参数用于计算单应矩阵的方法
        # 第四个参数为误差阈值
        # 返回值：H为变换矩阵,mask为掩膜
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # cv2.warpPerspective：利用H矩阵对原图进行简单的透视变换
        wrap = cv2.warpPerspective(img_left, H,
                                   (2 * w, 2 * h))
        wrap[0:h, 0:w] = img_right

        # 去除黑色无用部分
        rows, cols = np.where(wrap[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1

        match_result = wrap[min_row:max_row, min_col:max_col, :]
        return match_result


if __name__ == '__main__':
    pass
