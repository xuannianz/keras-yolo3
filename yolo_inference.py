from yolo import YOLO
from timeit import default_timer as timer
from yolo3.utils import resize_image
from PIL import Image
import numpy as np
import keras.backend as K
import glob
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# w, h
# model_image_size = (1920, 1216)
model_image_size = (416, 416)
yolo = YOLO(model_path='logs/2019-04-09/052-20.628-22.252.h5',
            anchors_path='model_data/voc_anchors.txt',
            classes_path='model_data/voc_classes.txt',
            model_image_size=model_image_size
            )

# image_paths = glob.glob('/home/adam/.keras/datasets/udacity_self_driving_car/object-dataset/*.jpg')
image_paths = glob.glob('/home/adam/.keras/datasets/VOCdevkit/test/VOC2007/JPEGImages/*.jpg')
num_classes = 20
colors = [np.random.randint(0, 256, 3).tolist() for i in range(num_classes)]
for image_path in image_paths:
    start = timer()
    image = Image.open(image_path)
    boxed_image = resize_image(image, model_image_size)
    image_data = np.array(boxed_image, dtype='float32')
    # (h, w, 3)
    # print(image_data.shape)
    image_data /= 255.
    # Add batch dimension.
    image_data = np.expand_dims(image_data, 0)
    # image.size 返回的是 (w, h), 那么 image_shape 就是 (h, w)
    image_shape = np.array([image.size[1], image.size[0]])
    image_shape = np.expand_dims(image_shape, 0)

    boxes, scores, classes = yolo.sess.run(
        [yolo.boxes, yolo.scores, yolo.classes],
        feed_dict={
            yolo.yolo_model.input: image_data,
            yolo.image_shape: image_shape,
            K.learning_phase(): 0
        })
    # 取第一个 batch_item
    boxes = boxes[0]
    scores = scores[0]
    classes = classes[0]
    filtered_boxes = boxes[scores > 0.0]
    filtered_classes = classes[scores > 0.0]
    filtered_scores = scores[scores > 0.0]
    print('Found {} boxes for image'.format(len(filtered_boxes)))
    image = np.array(image)[:, :, ::-1]
    image = image.copy()
    sorted_indexes = np.argsort(-filtered_scores)
    for i in sorted_indexes:
        class_id = int(filtered_classes[i])
        class_name = yolo.class_names[class_id]
        box = filtered_boxes[i]
        ymin = int(round(box[0]))
        xmin = int(round(box[1]))
        ymax = int(round(box[2]))
        xmax = int(round(box[3]))
        score = filtered_scores[i]
        label = '{} {:.2f}'.format(class_name, score)
        color = colors[class_id - 1]
        # ret[0] 表示包围 text 的矩形框的 width
        # ret[1] 表示包围 text 的矩形框的 height
        # baseline 表示的 text 最底下一个像素到文本 baseline 的距离
        # 文本 baseline 参考 https://blog.csdn.net/u010970514/article/details/84075776
        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.imshow('image', image)
    cv2.waitKey(0)
