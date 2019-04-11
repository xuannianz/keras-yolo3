import xml.etree.ElementTree as ET
import os.path as osp

sets = ([('2007', 'trainval'), ('2012', 'trainval')], [('2007', 'test')])

classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
           "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def convert_annotation(annotation_path, list_file):
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))


if __name__ == '__main__':
    for idx, image_sets in enumerate(sets):
        if idx == 0:
            voc_annotation_dir = '/home/adam/.keras/datasets/VOCdevkit/trainval'
            yolo_annotation_file = open(osp.join(voc_annotation_dir, 'train.txt'), 'w')
        else:
            voc_annotation_dir = '/home/adam/.keras/datasets/VOCdevkit/test'
            yolo_annotation_file = open(osp.join(voc_annotation_dir, 'test.txt'), 'w')
        for year, image_set in image_sets:
            image_set_file = open(osp.join(voc_annotation_dir, 'VOC{}/ImageSets/Main/{}.txt'.format(year, image_set)))
            image_ids = image_set_file.read().strip().split()
            images_dir = osp.join(voc_annotation_dir, 'VOC{}/JPEGImages'.format(year))
            annotations_dir = osp.join(voc_annotation_dir, 'VOC{}/Annotations'.format(year))
            for image_id in image_ids:
                image_path = osp.join(images_dir, '{}.jpg'.format(image_id))
                annotation_path = osp.join(annotations_dir, '{}.xml'.format(image_id))
                yolo_annotation_file.write(image_path)
                convert_annotation(annotation_path, yolo_annotation_file)
                yolo_annotation_file.write('\n')
        yolo_annotation_file.close()
