# Document-reconstruction

This project will consist of two parts: software and hardware.

## Software part

The purpose of the software part of this project:

1. **Image stitching**
2. **Text orientation correction in images**
3. *Framing text areas in images*
4. *OCR* (We will focus on OCR for handwritten Chinese and img2tex)
5. Text Correction (We will focus on Text Correction for Chinese)
6. Build the website and deploy the deep learning model to the cloud platform
7. more

## Hardware part

The purpose of the hardware part of this project:

1. Getting the images and remove them to the 
2. Transferring them to the device that performs the image processing
3. *Turning the pages of the book automatically*
4. more

> Completed functions are marked in bold.
> 
> Features that are already available but need refinement are marked in italics.
> 
> Others need to be pushed around by DDL / hhhfccz.

# How to do

I will introduce the overall project implementation idea according to two directions: software part and hardware part

## Software part

### Image stitching

**SIFT & knnMatch**, so easy, pass

you can see test samples in ['match_result'](https://github.com/hhhfccz/Document-reconstruction/tree/main/match_result)

### Text orientation correction in images

**Hough Linear Inspection & math**, so easy, pass

you can see test samples in ['rotated_result'](https://github.com/hhhfccz/Document-reconstruction/tree/main/rotated_result)

### Framing text areas in images

my idea is to use **MSER and NMS**, but you can see the result like:

![find_text_result2.jpg](https://raw.sevencdn.com/hhhfccz/Document-reconstruction/main/find_text_result/2.jpg)

It does not perform as well as expected.

So, i think Deep learning methods are the only ones that are truly appropriate for the task, like [CTPN](https://github.com/tianzhi0549/CTPN)

# How to use

make sure:

> opencv-python >= 4.4.0
> 
> opencv-python-contrib >= 4.4.0

OK! just do it!

> python main.py

so easy