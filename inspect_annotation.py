import numpy as np


train_annotation_path = \
    '/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/train_yolo.csv'
val_annotation_path = '/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/val_yolo.csv'
with open(train_annotation_path) as f:
    train_lines = f.readlines()
with open(val_annotation_path) as f:
    val_lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(train_lines)
np.random.shuffle(val_lines)
np.random.seed(None)

for i in range(len(val_lines)):
    line = val_lines[i]
    image_path = line.split(' ')[0]
    annotations = line.split(' ')[1:]
    for idx, annotation in enumerate(annotations):
        annotation = annotation.split(',')
        xmin = int(annotation[0])
        ymin = int(annotation[1])
        xmax = int(annotation[2])
        ymax = int(annotation[3])
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            print(w, h, xmin, ymin, xmax, ymax, image_path)
