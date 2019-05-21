import os
import numpy as np
import json
import tensorflow as tf
from glob import glob
from PIL import Image
import cv2
import skimage

# from text.keras_yolo3 import preprocess_true_boxes, yolo_text
Input = tf.keras.layers.Input
Lambda = tf.keras.layers.Lambda
load_model = tf.keras.models.load_model
Model = tf.keras.models.Model


def show_image(image):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', image)
    cv2.waitKey(0)


def polylines(img, points):
    im = np.zeros(img.shape[:2], dtype="uint8")
    for point in points:
        b = np.array([point], dtype=np.int32)
        cv2.fillPoly(im, b, 255)
    show_image(im)
    return im


def check_points(points, w, h):
    # 检测标注是否正确
    check = False
    for point in points:
        for x, y in point:
            if x > w or y > h:
                check = True
                break
        if check:
            break
    return check


def get_points(res):
    points = []
    for line in res:
        points.append(line['points'])
    return points


def resize_im(img, scale=416, max_scale=608):
    h, w = img.shape[:2]
    f = float(scale) / min(h, w)
    if max_scale is not None:
        if f * max(h, w) > max_scale:
            f = float(max_scale) / max(h, w)
    newW, newH = int(w * f), int(h * f)
    newW, newH = newW - (newW % 32), newH - (newH % 32)
    fw = w / newW
    fh = h / newH
    tmpImg = cv2.resize(img, None, None, fx=1 / fw, fy=1 / fh, interpolation=cv2.INTER_LINEAR)
    return tmpImg, fw, fh


def cleam_im(im):
    avg = 127
    im[im > avg] = 255
    im[im <= avg] = 0
    y, x = np.where(im == 255)
    xmin, ymin, xmax, ymax = (min(x), min(y), max(x), max(y))
    return xmin, ymin, xmax, ymax


def adjust_height(h):
    """
    调整 box 高
    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    N = len(heights)
    for i in range(N):
        if h <= heights[i] + heights[i] * 0.44 / 2:
            return heights[i]
    return h


def img_split_to_box(im, splitW=16, adjust=True):
    """
    均等分割box
    """
    tmpIm = im == 255
    h, w = tmpIm.shape[:2]
    num = w // splitW + 1

    box = []
    for i in range(num):
        xmin, ymin, xmax, ymax = splitW * i, 0, splitW * (i + 1), h
        ##迭代寻找最优ymin,ymax
        childIm = tmpIm[ymin:ymax, xmin:xmax]
        checkYmin = False
        checkYmax = False
        for j in range(ymax):
            if not checkYmin:
                if childIm[j].max():
                    ymin = j
                    checkYmin = True
            if not checkYmax:
                if childIm[ymax - j - 1].max():
                    ymax = ymax - j
                    checkYmax = True

        if adjust:
            childH = ymax - ymin + 1
            cy = (ymax + ymin) / 2
            childH = adjust_height(childH)
            ymin = cy - childH / 2
            ymax = cy + childH / 2
        box.append([xmin, ymin, xmax, ymax])

    return box


def resize_img_box(p, scale=416, max_scale=608, splitW=15, adjust=True):
    path = root.format(p)
    img = cv2.imread(path)
    if img is None:
        return None, []
    # train_labels, key 是 文件名去掉 .jpg, value 是一个 list, list 的每一个元素是一个 dict
    # dict 有两个 key, transcript 表示文本内容, points 表示文件框的坐标, language 表示文本的语种, illegibility 是否模糊
    points = get_points(train_labels[f'{p}'])
    h, w = img.shape[:2]
    check = check_points(points, w, h)
    if check:
        return None, []
    img, fw, fh = resize_im(img, scale=scale, max_scale=max_scale)
    boxes = []
    for point in points:
        # point 代表的是一个文本区域轮廓的点
        point = [[bx[0] / fw, bx[1] / fh] for bx in point]
        # point 组成的轮廓中的部分设置成 255, 其他设置成 0
        im = polylines(img, [point])
        # 说明 im 都是 0, 就是不包含文本区域
        if im.max() == 0:
            continue
        # 文本区域的 (xmin, ymin, xmax, ymax)
        xmin, ymin, xmax, ymax = cleam_im(im)
        tmp = im[ymin:ymax, xmin:xmax]
        box = img_split_to_box(tmp, splitW=splitW, adjust=adjust)
        childBoxes = []
        for bx in box:
            xmin_, ymin_, xmax_, ymax_ = bx

            xmin_, ymin_, xmax_, ymax_ = xmin + xmin_, ymin_ + ymin, xmax_ + xmin, ymax_ + ymin
            boxes.append([xmin_, ymin_, xmax_, ymax_])
            # childBoxes.append([xmin_,ymin_,xmax_,ymax_])
        # boxes.append(childBoxes)
    return img, boxes


def plot_box(img, boxes, color=(0, 0, 0)):
    tmp = np.copy(img)
    for box in boxes:
        cv2.rectangle(tmp, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 1)  # 19
    show_image(tmp)
    return Image.fromarray(tmp)


def clip_box(bbox, im_shape):
    # x1 >= 0
    bbox[:, 0] = np.maximum(np.minimum(bbox[:, 0], im_shape[1] - 1), 0)
    # y1 >= 0
    bbox[:, 1] = np.maximum(np.minimum(bbox[:, 1], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    bbox[:, 2] = np.maximum(np.minimum(bbox[:, 2], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    bbox[:, 3] = np.maximum(np.minimum(bbox[:, 3], im_shape[0] - 1), 0)

    return bbox


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def data_generator(roots, anchors, num_classes):
    n = len(roots)
    np.random.shuffle(roots)
    i = 0

    while True:
        p = roots[i]
        img, boxes = resize_img_box(p, scale=608, max_scale=1024, splitW=8, adjust=True)

        i += 1
        if i >= n:
            i = 0

        if img is None:
            continue

        h, w = img.shape[:2]
        input_shape = (h, w)
        boxes = np.array(boxes)
        # boxes = boxes[:,:4]
        boxes = clip_box(boxes, (h, w))
        newBox = np.zeros((len(boxes), 5))
        newBox[:, :4] = boxes
        newBox[:, 4] = 1
        boxes = newBox
        del newBox
        if np.random.randint(0, 100) > 70:
            if np.random.randint(0, 100) > 50:
                ##图像水平翻转
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

                img = cv2.flip(img, 1)
            else:
                ##垂直翻转
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]

                img = cv2.flip(img, 0)

        maxN = 128  ##随机选取128个box用于训练
        maxN = len(boxes)
        image, box = get_random_data(img, boxes, input_shape, max_boxes=maxN)

        image_data = np.array([image])
        box_data = np.array([box])
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        yield [image_data, *y_true], [np.zeros(1)] * 4


def split_contour(contour, offset_x, resized_width, offset_y, resized_height):
    """
    把 contour 包围的区别切分成 height 任意, weight 为 16 的矩形块
    Args:
        contour (np.array): shape 为 (m, 2) 表示 m 个 points 的坐标

    Returns:
        返回 np.array, shape 为 (n, 4) 表示 n 个矩形块的 (xmin, ymin, xmax, ymax) 坐标
    """
    boxes = []
    # np.array, (n, )
    blackboard = np.zeros((608, 608)).astype(np.uint8)
    rr, cc = skimage.draw.polygon(contour[:, 1].tolist(), contour[:, 0].tolist())
    if len(rr) == 0:
        return np.array(boxes)
    blackboard[rr, cc] = 255
    min_x = max(np.min(cc), offset_x)
    max_x = min(np.max(cc), offset_x + resized_width)
    min_y = max(np.min(rr), offset_y)
    max_y = min(np.max(rr), offset_y + resized_height)
    if max_x - min_x < 16:
        return np.array(boxes)

    max_x = max_x - ((max_x - min_x) % 16)
    this_min_y = min_y
    this_max_y = max_y
    for x in range(min_x + 16, max_x, 16):
        area = blackboard[:, x:x + 16]
        while not np.any(area[this_min_y]):
            this_min_y += 1
        while not np.any(area[this_max_y]):
            this_max_y -= 1
        boxes.append([x, this_min_y, x + 16, this_max_y])
        this_min_y = min_y
        this_max_y = max_y
    return np.array(boxes)


def convert_annotations(txt_annotation_path):
    txt_annotation_file = open(txt_annotation_path, 'w')
    for image_filename, labels in train_labels.items():
        # if image_filename != 'gt_2145':
        #     continue
        contours = []
        for label in labels:
            if label['illegibility']:
                continue
            else:
                contours.append(label['points'])
        image_path = image_path_template.format(image_filename)
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        scale = min(608 / image_width, 608 / image_height)
        resized_width = int(image_width * scale)
        resized_height = int(image_height * scale)
        resized_image = cv2.resize(image, (resized_width, resized_height))
        target_image = np.ones((608, 608, 3)).astype(np.uint8) * 128
        offset_x = (608 - resized_width) // 2
        offset_y = (608 - resized_height) // 2
        target_image[offset_y: offset_y + resized_height, offset_x: offset_x + resized_width] = resized_image
        resized_contours = []
        all_boxes = []
        for contour in contours:
            contour = np.array(contour)
            contour = contour * scale
            contour = np.round(contour)
            contour = contour.astype(np.int32)
            contour[:, 0] += offset_x
            contour[:, 1] += offset_y
            contour[:, 0] = np.clip(contour[:, 0], offset_x, offset_x + resized_width - 1)
            contour[:, 1] = np.clip(contour[:, 1], offset_y, offset_y + resized_height - 1)
            resized_contours.append(contour)
            boxes = split_contour(contour, offset_x, resized_width, offset_y, resized_height)
            if len(boxes) == 0:
                print(image_path)
                continue
            # for box in boxes:
            #     cv2.rectangle(target_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            all_boxes.extend(boxes.tolist())
        # resized_contours 可以是 shape 为 (m,n,2) 的 np.array, 也可以是 [(n,2),(n,2),(n,2)...(n,2)] 包含 np.array 的 list
        # resized_contours 不可以是嵌套的 list, TypeError: contours is not a numpy array, neither a scalar
        # cv2.drawContours(target_image, resized_contours, -1, (255, 255, 255), -1)
        box_annotations = [image_path]
        for box in all_boxes:
            # cv2.rectangle(target_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 1)
            # 0 表示 class_id
            box_annotations.append('{},{},{},{},0'.format(box[0], box[1], box[2], box[3]))
        txt_annotation_file.write(' '.join(box_annotations) + '\n')
        # show_image(target_image)


def show_origin_annotation(image_path, image_annotations):
    image = cv2.imread(image_path)
    contours = []
    for annotation in image_annotations:
        contours.append(np.array(annotation['points']))
    cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
    show_image(image)

if __name__ == '__main__':
    import os.path as osp

    dataset_dir = '/home/adam/.keras/datasets/icdar2019/ArT/'
    image_path_template = osp.join(dataset_dir, 'train_images/{}.jpg')
    with open(osp.join(dataset_dir, 'train_labels.json')) as f:
        train_labels = json.loads(f.read())
    convert_annotations('text/icdar_2019_art.txt')
    # show_origin_annotation(image_path_template.format('gt_2145'), train_labels['gt_2145'])
    # p = list(train_labels.keys())[32]
    # p = 'gt_183'
    # img, box = resize_img_box(p, scale=608, max_scale=1024, splitW=8, adjust=False)
    # plot_box(img, box, (0, 0, 255))
