# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/14
import cv2
import numpy as np
from simple_process import remove_the_bg


def find_local_maximums(num_rho, num_angle, threshold, accum):
    sort_buf = []
    for r in range(num_rho):
        for n in range(num_angle):
            base = (n + 1) * (num_rho + 2) + r + 1
            if accum[base] > threshold \
                    and accum[base] > accum[base - 1] \
                    and accum[base] >= accum[base + 1] \
                    and accum[base] > accum[base - num_rho - 2] \
                    and accum[base] >= accum[base + num_rho + 2]:
                sort_buf.append(base)
    return np.array(sort_buf).astype(np.int)


def create_trig_table(num_angle, min_theta, theta_step, i_rho):
    ang = float(min_theta)
    tab_sin = np.zeros(num_angle)
    tab_cos = np.zeros(num_angle)
    for i in range(num_angle):
        ang += float(theta_step)
        tab_sin[i] = float(np.sin(ang) * i_rho)
        tab_cos[i] = float(np.cos(ang) * i_rho)
    return tab_sin, tab_cos


def hough_lines_point_set(pts, lines_max=20, threshold=1,
                          min_rho=0.0, max_rho=360.0, rho_step=1,
                          min_theta=0.0, max_theta=np.pi / 2,
                          theta_step=np.pi / 180):
    # init
    i_rho = float(1 / rho_step)
    i_rho_min = float(min_rho * i_rho)
    num_angle = int(np.around((max_theta - min_theta) / theta_step))
    num_rho = int(np.around((max_rho - min_rho + 1) / rho_step))
    accum = np.zeros((num_angle + 10) * (num_rho + 10)).astype(np.int)
    # in opencv, +2, but it will throw IndexError,
    # it doesn't affect the final result

    # create sin and cos table
    tab_sin, tab_cos = create_trig_table(num_angle, min_theta, theta_step, i_rho)

    # stage 1, fill accumulator
    for i in range(len(pts)):
        for n in range(num_angle):
            r = int(np.around(pts[i, 1] * tab_cos[n] + pts[i, 0] * tab_sin[n] - i_rho_min))
            accum[(n + 1) * (num_rho + 2) + r + 1] += 1

    # stage 2, find local maximums
    sort_buf = find_local_maximums(num_rho, num_angle, threshold, accum)

    # stage 3, sort the detected lines by accumulator value
    z = zip(accum, sort_buf)
    z = sorted(z, reverse=True)
    accum, sort_buf = zip(*z)

    # stage 4, store the lines to the output buffer
    lines_max = np.min([lines_max, len(sort_buf)])
    scale = 1.0 / (num_rho + 2)
    line = {'votes': [], 'length_line': [], 'angle': []}
    for i in range(lines_max):
        idx = sort_buf[i]
        n = np.floor(idx * scale) - 1
        r = idx - (n + 1) * (num_rho + 2) - 1
        line['votes'].append(accum[i])
        line['length_line'].append(int(min_rho + r * rho_step))
        line['angle'].append(int((min_theta + n * theta_step) * 180 / np.pi))
    return line


def non_max_suppress(boxes, threshold=0.8):
    if len(boxes) == 0:
        return boxes
    else:
        boxes = boxes.astype("float32")
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        boxes_area = (x2 - x1 + 1) * (y2 - y1 + 1)
        # 排序
        idxs = np.argsort(y2)
        pick = []
        # 遍历重复框
        while len(idxs):
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
            idxs = np.delete(idxs, np.concatenate(([last], np.where(threshold < overlap)[0])))
        return boxes[pick].astype("int")


def detect(img_gray, norm=1.2):
    # 自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    img_gray = clahe.apply(img_gray)
    img_gray = remove_the_bg(img_gray)
    img_h, img_w = img_gray.shape

    # 创建mser实例
    mser = cv2.MSER_create(_delta=2, _min_area=300, _max_area=800, _max_variation=0.7)

    # 检测
    regions, _ = mser.detectRegions(img_gray)

    # 得到当前所有boxes
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    boxes = []
    for hull in hulls:
        x, y, w, h = cv2.boundingRect(hull)
        if w < norm * h and h < norm * w:
            boxes.append([x, y, x + w, y + h])
    # print(len(boxes))

    # 使用NMS算法抑制MSER检测效果
    boxes = non_max_suppress(np.array(boxes))

    # 找到boxes的中心点并连线
    pts = np.zeros((len(boxes), 2))
    pts[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    pts[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    pts = pts.astype(np.int)
    for pt in pts:
        # cv2.circle(img_gray, tuple(pt.astype(np.int).tolist()), 5, (0, 0, 0), 3)
        cv2.line(img_gray, (pt[0], 0), (pt[0], img_h), (0, 0, 0), 2)
        cv2.line(img_gray, (0, pt[1]), (img_w, pt[1]), (0, 0, 0), 2)
    # 利用霍夫变换查找点集中的直线，返回角度单位为弧度
    # lines = hough_lines_point_set(pts, min_theta=0., max_theta=np.pi/6)
    # print(lines)
    return img_gray


if __name__ == '__main__':
    pass
