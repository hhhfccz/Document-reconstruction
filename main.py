# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/8/29
import cv2
from rotation_according_to_word_direction import rotated_img_with_fft
from image_matching import get_match_img
from find_the_text import detect
from remove_the_background import remove_the_bg
from ocr_pytesseract import ocr


def main():
    print("please make sure there is a repeat between the two images.")
    # 输入接口（有点小问题，有空再改吧）
    number = input("choose your test number: ")
    img_right = cv2.imread("./pic_left/" + str(number) + ".jpg")
    img_left = cv2.imread("./pic_right/" + str(number) + ".jpg")

    print("----start the image processing----\n")

    # 将两个图片拼接在一起
    print("----matching----")
    _, result = get_match_img(img_left, img_right, norm=0.75)

    # 利用FFT进行文字方向矫正，使得文字正向
    print("----rotating----\n")
    img_gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    rotated = rotated_img_with_fft(result, img_gray)
    cv2.imwrite("./rotated_result/result" + str(number) + ".jpg", rotated)

    # 利用形态学找到文字大致范围并框选出来
    print("----finding the text----\n")
    text_area = detect(number, rotated)
    print("----picture processing, finished----\n")
    cv2.imwrite("./find_text_result/result" + str(number) + ".jpg", text_area)
    cv2.imshow("find text result", remove_the_bg(text_area))
    cv2.waitKey(0)

    # # 开始进行OCR
    # print("----ocr---")
    # ocr(number)
    print("----Done----")


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
