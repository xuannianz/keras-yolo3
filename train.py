"""
Retrain the YOLO model for your own dataset.
"""

import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from yolo3.model import preprocess_gt_boxes, yolo_body, tiny_yolo_body, yolo_loss
from yolo3.utils import get_random_data
import os
from datetime import date
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def _main():
    train_annotation_path = \
        '/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/train_yolo.csv'
    val_annotation_path = '/home/adam/.keras/datasets/udacity_self_driving_car/object-detection-crowdai/val_yolo.csv'
    log_dir = 'logs/{}'.format(str(date.today()))
    if not osp.exists(log_dir):
        os.mkdir(log_dir)
    classes_path = 'model_data/adam_classes.txt'
    anchors_path = 'model_data/yolo_anchors_adam.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)
    # multiple of 32, hw
    input_shape = (1216, 1920)

    is_tiny_version = len(anchors) == 6
    if is_tiny_version:
        model = create_tiny_model(input_shape, anchors, num_classes,
                                  freeze_body=2,
                                  weights_path='model_data/tiny_yolo_weights.h5')
    else:
        model = create_model(input_shape, anchors, num_classes,
                             # make sure you know what you freeze
                             freeze_body=2,
                             weights_path='model_data/yolo_weights.h5')

    csv_logger = CSVLogger(filename='yolo3_udacity_1216_1920_training_log.csv',
                           separator=',',
                           append=True)
    tensor_board = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss',
                                 # save_weights_only=True,
                                 save_best_only=True,
                                 # period=3
                                 )
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)

    # val_split = 0.1
    # with open(annotation_path) as f:
    #     lines = f.readlines()
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)
    # num_val = int(len(lines) * val_split)
    # num_train = len(lines) - num_val

    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    with open(val_annotation_path) as f:
        val_lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(train_lines)
    np.random.shuffle(val_lines)
    np.random.seed(None)
    num_train = len(train_lines)
    num_val = len(val_lines)

    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a not bad model.
    if True:
        # use custom yolo_loss Lambda layer.
        model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        # batch_size = 32
        batch_size = 4
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=50,
                            initial_epoch=0,
                            callbacks=[tensor_board, csv_logger, checkpoint])
        model.save_weights(log_dir + 'trained_weights_stage_1.h5')

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is not good.
    if True:
        print('Unfreeze all of the layers.')
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        # recompile to apply the change
        model.compile(optimizer=Adam(lr=1e-4), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
        # Note that more GPU memory is required after unfreezing the body
        # batch_size = 32
        batch_size = 1
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(data_generator_wrapper(train_lines, batch_size, input_shape, anchors, num_classes),
                            steps_per_epoch=max(1, num_train // batch_size),
                            validation_data=data_generator_wrapper(val_lines, batch_size, input_shape, anchors,
                                                                   num_classes),
                            validation_steps=max(1, num_val // batch_size),
                            epochs=100,
                            initial_epoch=50,
                            callbacks=[tensor_board, csv_logger, checkpoint, reduce_lr, early_stopping])
        model.save_weights(log_dir + 'trained_weights_final.h5')

    # Further training if needed.


def get_classes(classes_path):
    """loads the classes form a file"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def create_model(input_shape, anchors, num_classes,
                 load_pretrained=True,
                 freeze_body=2,
                 weights_path='model_data/yolo_weights.h5'):
    """
    create the training model

    Args:
        input_shape:
        anchors:
        num_classes:
        load_pretrained:
        freeze_body:
        weights_path:

    Returns:
    """

    # get a new session
    K.clear_session()
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    # dtype 默认为 float32
    y_true = [
        Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l], num_anchors // 3, num_classes + 5)) for
        l in range(3)]

    model_body = yolo_body(image_input, num_anchors // 3, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        # freeze_body=1 表示 freeze darknet-53 的部分
        # freeze_body=2 表示 freeze 除三个 prediction layers 之外的所有 layer
        if freeze_body in [1, 2]:
            # Freeze darknet53 body or freeze all but 3 output layers.
            # model_body.layers[185] 是 conv2d_53, 之前的部分都属于 darknet-53
            # model_body.layers[-1] 是 conv2d_75, 52*52 的 feature_map
            # model_body.layers[-2] 是 conv2d_67, 26*26 的 feature_map
            # model_body.layers[-3] 是 conv2d_59, 13*13 的 feature_map
            num = (185, len(model_body.layers) - 3)[freeze_body - 1]
            for i in range(num):
                model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss,
                        output_shape=(1,),
                        name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def create_tiny_model(input_shape, anchors, num_classes, load_pretrained=True, freeze_body=2,
                      weights_path='model_data/tiny_yolo_weights.h5'):
    """create the training model, for Tiny YOLOv3"""
    K.clear_session()  # get a new session
    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    num_anchors = len(anchors)

    y_true = [Input(shape=(h // {0: 32, 1: 16}[l], w // {0: 32, 1: 16}[l], \
                           num_anchors // 2, num_classes + 5)) for l in range(2)]

    model_body = tiny_yolo_body(image_input, num_anchors // 2, num_classes)
    print('Create Tiny YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    if load_pretrained:
        model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print('Load weights {}.'.format(weights_path))
        if freeze_body in [1, 2]:
            # Freeze the darknet body or freeze all but 2 output layers.
            num = (20, len(model_body.layers) - 2)[freeze_body - 1]
            for i in range(num): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.7})(
        [*model_body.output, *y_true])
    model = Model([model_body.input, *y_true], model_loss)

    return model


def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """
    data generator for fit_generator

    Args:
        annotation_lines:
        batch_size:
        input_shape:
        anchors (np.array): (num_anchors, 2)
        num_classes:

    Returns:

    """
    n = len(annotation_lines)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(annotation_lines)
            image, box = get_random_data(annotation_lines[i], input_shape, random=False)
            # image, box = get_random_data(annotation_lines[i], input_shape, random=True)
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        # y_true 一个 list, 每一个元素表示一个 output_layer 的一个 batch 的 y_true
        y_true = preprocess_gt_boxes(box_data, input_shape, anchors, num_classes)
        # np.zeros(batch_size) 表示 model 的 output, 对应 yolo_loss
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(annotation_lines, batch_size, input_shape, anchors, num_classes):
    """

    Args:
        annotation_lines:
        batch_size:
        input_shape:
        anchors (np.array): (num_anchors,2)
        num_classes:

    Returns:

    """
    n = len(annotation_lines)
    if n == 0 or batch_size <= 0:
        raise ValueError('`annotation_lines` cannot be empty and `batch_size` must be > 0')
    return data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
