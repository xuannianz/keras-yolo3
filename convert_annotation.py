import xml.etree.ElementTree as ET
import glob
import os.path as osp
import logging
import sys
import csv

logger = logging.getLogger('xml2csv')
logger.setLevel(logging.DEBUG)  # default log level
formatter = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
sh.setFormatter(formatter)
logger.addHandler(sh)

classes = ['no', 'date']


def convert_annotation(xml_annotation_path, txt_file_obj):
    xml_file_obj = open(xml_annotation_path)
    tree = ET.parse(xml_file_obj)
    root = tree.getroot()
    image_path = root.find('path').text
    txt_file_obj.write(image_path)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        txt_file_obj.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    txt_file_obj.write('\n')
    xml_file_obj.close()


# annotations_dir = '/home/adam/Pictures/invoice/annotations_0309'
# txt_annotation_path = osp.join(annotations_dir, 'train.txt')
# txt_file_obj = open(txt_annotation_path, 'w')
# for xml_annotation_path in glob.glob(osp.join(annotations_dir, '*.xml')):
#     logger.debug('converting {}'.format(xml_annotation_path))
#     convert_annotation(xml_annotation_path, txt_file_obj)
# txt_file_obj.close()

def convert_self_driving(annotation_path):
    images_dir = osp.split(annotation_path)[0]
    annotation_file = open(annotation_path, 'r')
    csv_reader = csv.reader(annotation_file, delimiter=',')
    annotations = {}
    # class_names = ('Car', 'Truck', 'Pedestrian')
    for row in csv_reader:
        image_filename = row[0]
        xmin = row[1]
        ymin = row[2]
        xmax = row[3]
        ymax = row[4]
        # 0 不再是 background
        class_id = str(int(row[5]) - 1)
        if image_filename not in annotations:
            image_path = osp.join(images_dir, image_filename)
            annotations[image_filename] = [image_path]
        annotations[image_filename].append(','.join([xmin, ymin, xmax, ymax, class_id]))
    annotation_file.close()
    new_annotation_path = osp.splitext(annotation_path)[0] + '_yolo.csv'
    new_annotation_file = open(new_annotation_path, 'w')
    csv_writer = csv.writer(new_annotation_file, delimiter=' ')
    for image_filename in annotations:
        annotation = annotations[image_filename]
        csv_writer.writerow(annotation)
    new_annotation_file.close()


convert_self_driving('/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/train.csv')
