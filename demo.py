# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/8/29
import cv2
from img_rotate.image_rotate import rotate_img
from img_stitch.image_match import get_match_img
from layout_analysis.text_find import detect, get_roi
from utils.used_time import decorator_used_time
from utils.img_preprocess import cv_show


@decorator_used_time
def main():
    print("please make sure there is a repeat between the two images.")

    img_right = cv2.imread("path of left img")
    img_left = cv2.imread("path of right img")

    print("----start the image processing----")

    # image stitching
    print("----stitching----")
    img_match = get_match_img(img_left, img_right)
    cv_show(img_match)

    # image rotating
    print("----rotating----")
    img_rotated = rotate_img(img_match)
    cv_show(img_rotated)

    # layout analysis
    print("----finding the text----")
    SCREEN_OR_NOT = input("Add outliers-screening or not: y/N ")
    text_area, pts = detect(img_rotated, norm=2, SCREEN_OR_NOT=SCREEN_OR_NOT)
    img_text_area = get_roi(img_rotated, text_area, pts)
    cv_show(img_text_area)

    print("----Done----")


if __name__ == "__main__":
    main()
