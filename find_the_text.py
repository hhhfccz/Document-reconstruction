# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/14
import cv2
import numpy as np
from get_roi import get_roi


def non_max_suppress(boxes, threshold=0.8):
    if len(boxes) == 0:
        return boxes
    else:
        boxes = boxes.astype("float32")
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes_area = (x2 - x1 + 1)*(y2 - y1 + 1)
        # 排序
        idxs = np.argsort(y2)
        pick = []
        # 遍历重复框
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # 找剩下的其余框中最大坐标和最小坐标
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # 计算重叠面积占对应框的比例，即 IoU
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / boxes_area[idxs[:last]]

            # 剔除
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > threshold)[0])))
        return boxes[pick].astype("int")


def detect(img_gray, norm=1.2):
    # 创建mser实例
    mser = cv2.MSER_create(_delta=2, _min_area=300, _max_area=800, _max_variation=0.7)

    # 检测
    regions, _ = mser.detectRegions(img_gray)

    # 得到文本区域
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

    # 得到当前所有boxes
    boxes = []
    for c in hulls:
        x, y, w, h = cv2.boundingRect(c)
        boxes.append([x, y, x + w, y + h])
        if w < norm * h and h < norm * w:
            cv2.rectangle(img_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # boxes = non_max_suppress(np.array(boxes), threshold=0.5)

    # for (x, y, w, h) in boxes:
    #     if w < norm * h and h < norm * w:
    #         cv2.rectangle(img_gray, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img_gray


if __name__ == '__main__':
    pass
