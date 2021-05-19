# ——*——coding:utf-8——*——
# author: hhhfccz(胡珈魁) time:2020/11/14
import cv2
import numpy as np
from utils.img_preprocess import remove_bg
from sklearn.ensemble import IsolationForest


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


def corner_detect(img):
    img_new = np.zeros_like(img)
    corners = cv2.goodFeaturesToTrack(img, 10000, 0.01, 10)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img_new, (x, y), 3, 255, -1)
    return img_new


def level_map(img_gray):
    """
    TODO: this function is so ugly, i should update it later
    """
    h, w = img_gray.shape
    map_ans = []
    for i in range(h):
        if np.array([img_gray[i, :] == 0]).any():
            map_ans.append(i)
    # print(map_ans)

    # 判断是否存在连续序列，并初步提取
    lists_mapping = []
    list_mapping = []
    for i in range(1, len(map_ans)):
        a = map_ans[i]
        b = map_ans[i-1] + 1
        if abs(a - b) <= 10:
            list_mapping.append(b - 1)
        else:
            list_mapping = []
        if len(lists_mapping) == 0 or (len(lists_mapping) != 0 and list_mapping != lists_mapping[-1]):
            lists_mapping.append(list_mapping)

    # 去重
    lists_mapping_new = []
    for list_mapping in lists_mapping:
        if list_mapping not in lists_mapping_new:
            lists_mapping_new.append(list_mapping)
    return lists_mapping_new


def get_roi(img_gray, lists_mapping, pts):
    for list_mapping in lists_mapping:
        h_min = list_mapping[0]
        h_max = list_mapping[-1]

        # 检测行范围内pts的存在，并以存在的pts左右两端为文本行左右边界
        pts_w = []
        pts_h = pts[:, 1]
        for i in range(len(pts_h)):
            pt_h = pts_h[i]
            if h_min-10 <= pt_h <= h_max+10:
                pts_w.append(pts[i, 0])

        pt1 = np.array([min(pts_w) - 10, h_min - 10], np.int).tolist()
        pt2 = np.array([max(pts_w) + 10, h_max + 10], np.int).tolist()
        cv2.rectangle(img_gray, tuple(pt1), tuple(pt2), 0, 5)
    return img_gray


def get_whole_roi(img_gray, pts, img_w, img_h):
    # get the text roi
    img_new_1 = np.ones_like(img_gray) * 255
    img_new_2 = np.ones_like(img_gray) * 255
    for pt in pts:
        cv2.line(img_new_1, (0, pt[1]), (img_w, pt[1]), 0, 1)
        cv2.line(img_new_2, (pt[0], 0), (pt[0], img_h), 0, 1)
    img_new_1 = cv2.morphologyEx(img_new_1, cv2.MORPH_OPEN, np.ones((5, 5)))
    img_new_2 = cv2.morphologyEx(img_new_2, cv2.MORPH_OPEN, np.ones((5, 5)))
    img_new = img_new_1 + img_new_2
    return img_new


def screen_outliers(pts):
    # 随机森林剔除异常点
    model_iforest = IsolationForest(n_estimators=100,
                                    max_samples="auto",
                                    contamination=0.1,
                                    max_features=0.1)
    model_iforest.fit(pts)
    exception_pts_anomaly = model_iforest.predict(pts)
    pts_new = []
    [pts_new.append(pts[i]) for i in range(len(pts)) if exception_pts_anomaly[i] == 1]
    return np.array(pts_new, np.int)


def detect(img_gray, norm=1.2, SCREEN_OR_NOT=0):
    # 自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    img_gray = clahe.apply(img_gray)
    img_gray = remove_bg(img_gray)
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

    # 找到boxes的中心点
    pts = np.zeros((len(boxes), 2))
    pts[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2
    pts[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
    pts = pts.astype(np.int)

    # 随机森林剔除异常点，可选
    if SCREEN_OR_NOT == "y":
        pts = screen_outliers(pts)
    elif SCREEN_OR_NOT == "N":
        print("We will not screen the outliers.")

    # get the text roi
    img_text_whole_roi = get_whole_roi(img_gray, pts, img_w, img_h)

    # get the lines
    lists_mapping = level_map(img_text_whole_roi)
    lists_mapping_new = []
    for list_mapping in lists_mapping:
        if len(list_mapping) >= 1:
            lists_mapping_new.append(list_mapping)
    # print(len(lists_mapping_new))

    return lists_mapping_new, pts


if __name__ == '__main__':
    pass
