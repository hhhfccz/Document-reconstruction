# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/19
import cv2
import pytesseract
import os


def ocr(img_number):
    # init
    pytesseract.pytesseract.tesseract_cmd = 'D:\\Tesseract-OCR\\tesseract.exe'
    # ocr
    img_path = "./get_text_roi/img" + str(img_number)
    img_files = os.listdir(img_path)
    # 其中img_number为第几组图像
    for img_file in img_files:
        img = cv2.imread(img_path + "/" + img_file)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(pytesseract.image_to_string(img_rgb))
