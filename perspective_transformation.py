# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/12/1
import numpy as np
import cv2


def getkpoint(imag, input1):
    mask1 = np.zeros_like(input1)
    x = 0
    y = 0
    w1, h1 = input1.shape
    input1 = input1[0:w1, 200:h1]

    try:
        w, h = imag.shape
    except:
        return None

    mask1[y:y + h, x:x + w] = 255  # 整张图片像素
    keypoint = []
    kp = cv2.goodFeaturesToTrack(input1, 200, 0.04, 7)
    if kp is not None and len(kp) > 0:
        for x, y in np.float32(kp).reshape(-1, 2):
            keypoint.append((x, y))
    return keypoint


def process(image):
    grey1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    grey = cv2.equalizeHist(grey1)
    keypoint = getkpoint(grey, grey1)

    if keypoint is not None and len(keypoint) > 0:
        for x, y in keypoint:
            cv2.circle(image, (int(x + 200), y), 3, (255, 255, 0))
    return image


import cv2
import numpy as np


def get_contour(img):
    """获取连通域
    :param img: 输入图片
    :return: 最大连通域
    """
    ret, img_bin = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        print("轮廓 %d 的面积是:%d" % (i, area))

        areas.append(area)
    index = np.argmax(areas)

    return img_bin, contours[index]


def get_cornerHarris(img_src):
    """
        获取图像角点
    :param img_src: 处理图像
    :return: 角点图像
    """
    img_corner = np.zeros(img_src.shape, np.uint8)
    img_gray = img_src.copy()
    img_gray = np.float32(img_gray)

    img_dist = cv2.cornerHarris(img_gray, 5, 5, 0.04)
    img_dist = cv2.dilate(img_dist, None)

    img_corner[img_dist > 0.01 * img_dist.max()] = [255]

    return img_corner


def get_warpPerspective(points, image_src):
    """
    执行透视变换
    :param points: 输入的四个角点
    :param image_src: 输入的图片
    :return: 变换后的图片
    """
    src_point = np.float32([
        [points[2][0], points[2][1]],
        [points[3][0], points[3][1]],
        [points[1][0], points[1][1]],
        [points[0][0], points[0][1]]])
    width = 1920
    height = 1080
    dst_point = np.float32([[0, 0], [width - 1, 0],
                            [0, height - 1], [width - 1, height - 1]])

    perspective_matrix = cv2.getPerspectiveTransform(src_point, dst_point)

    img_dst = cv2.warpPerspective(image_src, perspective_matrix, (width, height))
    return img_dst


def main():
    # 读取图片
    img_white = cv2.imread("E:\\cv\\Document-reconstruction\\pic_left\\2.jpg", cv2.IMREAD_GRAYSCALE)
    img_book = cv2.imread("E:\\cv\\Document-reconstruction\\pic_right\\2.jpg", cv2.IMREAD_GRAYSCALE)

    cv2.imshow("img", img_book)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 最大的轮廓
    img_bin, contour = get_contour(img_white)

    # 处理区域mask
    mask = np.zeros(img_white.shape, np.uint8)
    mask = cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    # resize_show_image("img_mask", mask)

    # 获取区域角点图片
    img_corner = get_cornerHarris(img_white)

    # 膨胀和 mask与角点操作
    kernel = np.ones((5, 5), np.uint8)
    img_mask = cv2.dilate(mask, kernel)
    img_corner = cv2.bitwise_and(img_mask, img_corner)
    # cv2.imshow("cornor“, img_corner)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # 获取四个角点的中心坐标
    contours, hierarchy = cv2.findContours(img_corner, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for i in range(len(contours)):
        center, radius = cv2.minEnclosingCircle(contours[i])
        points.append(center)
    print(points)

    # 透视变换
    img_dst = get_warpPerspective(points, img_book)
    cv2.imshow("image_dst", img_dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # img = cv2.imread("E:\\cv\\Document-reconstruction\\rotated_result\\result2.jpg")
    # img = process(img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    main()
