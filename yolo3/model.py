"""YOLO_v3 Model Defined in Keras."""
from functools import wraps
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from yolo3.utils import compose
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


@wraps(Conv2D)
def darknet_conv2d(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = dict({'kernel_regularizer': l2(5e-4)})
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides') == (2, 2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def darknet_conv2d_bn_leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        darknet_conv2d(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def resblock_body(x, num_filters, num_blocks):
    """A series of resblocks starting with a downsampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(x)
    x = darknet_conv2d_bn_leaky(num_filters, (3, 3), strides=(2, 2))(x)
    for i in range(num_blocks):
        y = compose(
            darknet_conv2d_bn_leaky(num_filters // 2, (1, 1)),
            darknet_conv2d_bn_leaky(num_filters, (3, 3)))(x)
        x = Add()([x, y])
    return x


def darknet_body(x):
    """Darknent body having 52 Convolution2D layers"""
    x = darknet_conv2d_bn_leaky(32, (3, 3))(x)
    x = resblock_body(x, 64, 1)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    x = resblock_body(x, 512, 8)
    x = resblock_body(x, 1024, 4)
    return x


def make_last_layers(x, num_filters, out_filters):
    """6 Conv2D_BN_Leaky layers followed by a Conv2D_linear layer"""
    x = compose(darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)),
                darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d_bn_leaky(num_filters, (1, 1)))(x)
    y = compose(darknet_conv2d_bn_leaky(num_filters * 2, (3, 3)),
                darknet_conv2d(out_filters, (1, 1)))(x)
    return x, y


def yolo_body(inputs, num_anchors, num_classes):
    """Create YOLO_V3 model CNN body in Keras."""
    darknet = Model(inputs, darknet_body(inputs))
    # x 是 leaky_re_lu_57, y1 是 conv2d_59, 13*13
    x, y1 = make_last_layers(darknet.output, 512, num_anchors * (num_classes + 5))
    x = compose(darknet_conv2d_bn_leaky(256, (1, 1)),
                UpSampling2D(2))(x)
    # darknet[152] 是 add_19
    x = Concatenate()([x, darknet.layers[152].output])
    # x 是 leaky_re_lu_64, y2 是 conv2d_67, 26*26
    x, y2 = make_last_layers(x, 256, num_anchors * (num_classes + 5))
    x = compose(darknet_conv2d_bn_leaky(128, (1, 1)),
                UpSampling2D(2))(x)
    # darknet[92] 是 add_11
    x = Concatenate()([x, darknet.layers[92].output])
    # x 是 leaky_re_lu_71, y3 是 conv2d_75, 52*52
    x, y3 = make_last_layers(x, 128, num_anchors * (num_classes + 5))

    return Model(inputs, [y1, y2, y3])


def tiny_yolo_body(inputs, num_anchors, num_classes):
    """Create Tiny YOLO_v3 model CNN body in keras."""
    x1 = compose(
        darknet_conv2d_bn_leaky(16, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        darknet_conv2d_bn_leaky(32, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        darknet_conv2d_bn_leaky(64, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        darknet_conv2d_bn_leaky(128, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        darknet_conv2d_bn_leaky(256, (3, 3)))(inputs)
    x2 = compose(
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        darknet_conv2d_bn_leaky(512, (3, 3)),
        MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'),
        darknet_conv2d_bn_leaky(1024, (3, 3)),
        darknet_conv2d_bn_leaky(256, (1, 1)))(x1)
    y1 = compose(
        darknet_conv2d_bn_leaky(512, (3, 3)),
        darknet_conv2d(num_anchors * (num_classes + 5), (1, 1)))(x2)

    x2 = compose(
        darknet_conv2d_bn_leaky(128, (1, 1)),
        UpSampling2D(2))(x2)
    y2 = compose(
        Concatenate(),
        darknet_conv2d_bn_leaky(256, (3, 3)),
        darknet_conv2d(num_anchors * (num_classes + 5), (1, 1)))([x2, x1])

    return Model(inputs, [y1, y2])


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """
    Convert final layer features to bounding box parameters.

    Args:
        feats: 一个 feature map 上的预测结果,
            (batch_size, f_height, f_width, num_anchors_per_output_layer * (5 + num_classes)
        anchors: 用于一个 feature map 上的 anchor, (num_anchors_this_layer, 4)
        num_classes:
        input_shape:
        calc_loss:

    Returns:

    """
    num_anchors_this_layer = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors_this_layer, 2])
    # feature_map 的 size, (f_height, f_width)
    grid_shape = K.shape(feats)[1:3]
    # K.arange 之后的 shape 为 (f_height, ), K.reshape 之后变为 (f_height, 1, 1, 1),
    # K.tile 之后变为 (f_height, f_width, 1, 1)
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, 1, 1])
    # Note K.concatenate 的 axis 默认为 -1 , 而 np.concatenate 的 axis 默认为 0
    # (f_height, f_width, 1, 2)
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))
    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors_this_layer, num_classes + 5])
    # Adjust predictions to each spatial grid point and anchor size.
    # feats 的 shape 为 (batch_size, f_height, f_width, num_anchors_this_layer, 2)
    # grid 的 shape 为 (f_height, f_width, 1, 2), 两者想加后的 shape 和 feats 相同
    # / 是用来 normalize
    # 公式参考原论文
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    """Get corrected boxes"""
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))
    # 对于 udacity self-driving dataset 来说, input_shape 为 (1216, 1920), image_shape 为 (1200, 1920)
    # new_shape (1200, 1920)
    new_shape = K.round(image_shape * K.min(input_shape / image_shape))
    # (1216-1200/2/1216, 0)
    # / input_shape 是 normalize, 后面的 box_yx 也是 normalize 过的
    offset = (input_shape - new_shape) / 2. / input_shape
    # (1216/1200, 1)
    # 现在要把 box_yx, box_hw 转成 new_shape 中的值, 注意它们并不相等, 因为两者并不是同比例缩放
    scale = input_shape / new_shape
    # Note: 为什么要减去 offset?
    # offset 的部分是填充部分, new_shape 中是不包含这部分的
    # Note: 为什么要乘以 scale?
    # 举个例子, 假设 box_yx 为 (0.5, 0.5), offset 为 (0.1, 0), 那么 box_yx - offset = (0.4, 0.5)
    # box 的中心点在 input 中为中心点, 在 new_shape 中也应该是中心点, 而此时为 0.4, 要把它变成 0.5, 0.4 * (0.5 * 2 / 0.4 * 2)=0.5
    # scale = (0.5 * 2) / (0.4 * 2) 就是 input_shape / new_shape
    # 也就说 input 中的 0.4 相当于 new_shape 中的 0.5, 所有 box_h 也是要乘以这个比例的
    box_yx = (box_yx - offset) * scale
    box_hw *= scale
    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        # y_min
        box_mins[..., 0:1],
        # x_min
        box_mins[..., 1:2],
        # y_max
        box_maxes[..., 0:1],
        # x_max
        box_maxes[..., 1:2]
    ])

    # Scale boxes back to original image shape.
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape):
    """
    Process Conv layer output
    Args:
        feats:
        anchors: (num_anchors_this_layer, 2)
        num_classes:
        input_shape: (2, ) hw
        image_shape: (batch_size, 2)

    Returns:

    """
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    # adam for batch predictions
    batch_size = K.shape(image_shape)[0]
    input_shape = K.expand_dims(input_shape, axis=0)
    input_shape = K.tile(input_shape, (batch_size, 1))
    elems = (box_xy, box_wh, input_shape, image_shape)
    boxes = tf.map_fn(lambda x: yolo_correct_boxes(x[0], x[1], x[2], x[3]), elems=elems, dtype=tf.float32)
    box_scores = box_confidence * box_class_probs
    batch_size = tf.shape(feats)[0]
    boxes = tf.reshape(boxes, [batch_size, -1, 4])
    box_scores = tf.reshape(box_scores, [batch_size, -1, num_classes])
    return boxes, box_scores


def yolo_eval(yolo_outputs,
              anchors,
              num_classes,
              image_shape,
              max_boxes=20,
              score_threshold=.6,
              iou_threshold=.5,
              max_boxes_per_image=200,
              ):
    """
    Evaluate yolo model on given input and return filtered boxes.

    Args:
        yolo_outputs (list): [conv59_output, conv67_output, conv75_output]
        anchors (np.array): anchor 文件中的所有 anchors
        num_classes (int):
        image_shape (tensor): (2, ) (image_height, image_width)
        max_boxes: 一个 image 上属于某个 class 的最大的 boxes 个数
        score_threshold:
        iou_threshold:
        max_boxes_per_image: 一个 image 上最大 boxes 个数

    Returns:

    """
    num_output_layers = len(yolo_outputs)
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_output_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    # tensor, (2, )
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    # 每一个元素是一个 tensor (batch_size, num_boxes_this_layer, 4), 表示某一输出层上所有的 boxes
    boxes_per_output_layer = []
    scores_per_output_layer = []
    for l in range(num_output_layers):
        output_layer_boxes, output_layer_scores = yolo_boxes_and_scores(yolo_outputs[l],
                                                                        anchors[anchor_mask[l]],
                                                                        num_classes,
                                                                        input_shape,
                                                                        image_shape,
                                                                        )
        boxes_per_output_layer.append(output_layer_boxes)
        scores_per_output_layer.append(output_layer_scores)

    # adam for batch inference, axis=0 表示的是 batch 那一维, axis=1 表示的是 boxes_per_layer 那一维
    boxes = K.concatenate(boxes_per_output_layer, axis=1)
    scores = K.concatenate(scores_per_output_layer, axis=1)

    mask = scores >= score_threshold
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    max_boxes_per_image_tensor = K.constant(max_boxes_per_image, dtype='int32')

    # 每一个元素是一个 tensor 表示一个 batch_item 的预测值
    # 用 list 而不是 tensor 来存储是因为每一个 batch_item 预测到的 boxes 的数量经过过滤后会不一样
    boxes_list = []
    scores_list = []
    class_ids_list = []
    # 返回一个 list, 每一个元素表示一个 batch_item 上未过滤的 boxes

    # def loop_body(x, i_, batch_boxes_):
    #     batch_item_boxes_ = tf.gather(x, i)
    #     batch_boxes_ = tf.concat([batch_boxes_, [batch_item_boxes_]], 0)
    #     return x, i_ + 1, batch_boxes_
    # batch_boxes = tf.Variable([])
    # i = tf.constant(0)
    # _, _, batch_boxes = tf.while_loop(lambda x, i_, _: i_ < batch_size,
    #                                   loop_body,
    #                                   (boxes, 0, batch_boxes),
    #                                   shape_invariants=(boxes.get_shape(), i.get_shape(), tf.TensorShape(None)))
    # batch_boxes = tf.unstack(boxes)
    # partitioned = tf.dynamic_partition(boxes, tf.range(batch_size), batch_size, name='dynamic_unstack')
    # batch_scores = tf.unstack(scores)
    # batch_mask = tf.unstack(mask)
    def evaluate_batch_item(batch_item_boxes, batch_item_scores, batch_item_mask):
        # 每一个元素表示一个 batch_item 上属于某一个类的 boxes
        boxes_per_class = []
        scores_per_class = []
        class_ids_per_class = []
        for c in range(num_classes):
            # TODO: use keras backend instead of tf.
            class_boxes = tf.boolean_mask(batch_item_boxes, batch_item_mask[:, c])
            class_scores = tf.boolean_mask(batch_item_scores[:, c], batch_item_mask[:, c])
            nms_index = tf.image.non_max_suppression(class_boxes,
                                                     class_scores,
                                                     max_boxes_tensor,
                                                     iou_threshold=iou_threshold)
            class_boxes = K.gather(class_boxes, nms_index)
            class_scores = K.gather(class_scores, nms_index)
            class_class_ids = K.ones_like(class_scores, 'float32') * c
            boxes_per_class.append(class_boxes)
            scores_per_class.append(class_scores)
            class_ids_per_class.append(class_class_ids)
        filtered_batch_item_boxes = K.concatenate(boxes_per_class, axis=0)
        filtered_batch_item_scores = K.concatenate(scores_per_class, axis=0)
        filtered_batch_item_scores = K.expand_dims(filtered_batch_item_scores, axis=-1)
        filtered_batch_item_class_ids = K.concatenate(class_ids_per_class, axis=0)
        filtered_batch_item_class_ids = K.expand_dims(filtered_batch_item_class_ids, axis=-1)
        filtered_batch_item_predictions = K.concatenate([filtered_batch_item_boxes,
                                                         filtered_batch_item_scores,
                                                         filtered_batch_item_class_ids], axis=-1)
        batch_item_num_predictions = tf.shape(filtered_batch_item_boxes)[0]
        padded_batch_item_predictions = tf.pad(tensor=filtered_batch_item_predictions,
                                               paddings=[[0, max_boxes_per_image_tensor - batch_item_num_predictions],
                                                         [0, 0]],
                                               mode='CONSTANT',
                                               constant_values=0.0)
        return padded_batch_item_predictions

    predictions = tf.map_fn(lambda x: evaluate_batch_item(x[0], x[1], x[2]),
                            elems=(boxes, scores, mask),
                            dtype=tf.float32)

    return predictions[..., :4], predictions[..., 4], predictions[..., 5]
    # return boxes_list, scores_list, class_ids_list


def preprocess_gt_boxes(gt_boxes, input_shape, anchors, num_classes):
    """
    Preprocess ground truth boxes to model input format

    首先为每个 gt_box 寻找到最大 iou 的 anchor
    在根据 gt_box 的 normalize 过的中心点坐标在 anchor 所在的 feature_map 找到对应的位置, 设置相应的值

    Args:
        gt_boxes (np.array): (batch_size, max_boxes, 5)
            Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
            如果一个 image 上的 gt_box 数量不足 max_boxes, 那么填充了 0
        input_shape (tuple): (h, w) multiples of 32
        anchors (np.array):  shape=(num_anchors, 2), wh
        num_classes (int):

    Returns:
        y_true: list of array, shape like yolo_outputs, xywh are relative value
            每一个元素的 shape 为 (batch_size, f_h, f_w, num_anchors_per_output_layer, num_classes + 5)

    """
    assert (gt_boxes[..., 4] < num_classes).all(), 'class id must be less than num_classes'

    num_output_layers = len(anchors) // 3
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_output_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    # gt_boxes = np.array(gt_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # 中心点坐标
    boxes_xy = (gt_boxes[..., 0:2] + gt_boxes[..., 2:4]) // 2
    boxes_wh = gt_boxes[..., 2:4] - gt_boxes[..., 0:2]
    # normalize
    gt_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
    gt_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

    batch_size = gt_boxes.shape[0]
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_output_layers)]
    y_true = [np.zeros((batch_size, grid_shapes[l][0], grid_shapes[l][1], len(anchor_mask[l]), 5 + num_classes),
                       dtype='float32') for l in range(num_output_layers)]

    # Expand dim to apply broadcasting.
    # (1, num_anchors, 2)
    anchors = np.expand_dims(anchors, 0)
    # 以中心点位置为 (0, 0), 计算 anchor 和 gt_box 的 iou
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    # FIXME: 删除之前为了满足 max_boxes 而做的填充
    valid_mask = boxes_wh[..., 0] > 0

    for b in range(batch_size):
        # Discard zero rows.
        # (num_non_zero_boxes, 2)
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh) == 0:
            continue
        # Expand dim to apply broadcasting.
        # (num_non_zero_boxes, 1, 2)
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        # (num_non_zero_boxes, num_anchors, 2)
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        # (num_non_zero_boxes, num_anchors)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        # (num_non_zero_boxes, 1)
        box_area = wh[..., 0] * wh[..., 1]
        # (1, num_anchors)
        anchor_area = anchors[..., 0] * anchors[..., 1]
        # box_area + anchor_area 的 shape 为 (num_non_zero_boxes, num_anchors)
        iou = intersect_area / (box_area + anchor_area - intersect_area)

        # Find best anchor for each gt box
        best_anchor_ids = np.argmax(iou, axis=-1)

        for gt_box_id, best_anchor_id in enumerate(best_anchor_ids):
            for l in range(num_output_layers):
                if best_anchor_id in anchor_mask[l]:
                    # gt_box 在 feature map 中的坐标, i 表示横坐标, j 表示纵坐标
                    i = np.floor(gt_boxes[b, gt_box_id, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(gt_boxes[b, gt_box_id, 1] * grid_shapes[l][0]).astype('int32')
                    # k 表示在 best_anchor_id 在 anchor_mask 的位置
                    k = anchor_mask[l].index(best_anchor_id)
                    # c 表示 class_id
                    c = gt_boxes[b, gt_box_id, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = gt_boxes[b, gt_box_id, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5 + c] = 1
    return y_true


def box_iou(b1, b2):
    """
    Return iou tensor
    Args:
        b1 (tensor): (..., 4)
        b2 (tensor): (num_b2_boxes, 4)

    Returns:
        iou (tensor): shape=(num_b1_boxes, num_b2_boxes)
    """
    # Expand dim to apply broadcasting.
    # (..., 1, 4)
    b1 = K.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    # (1, num_b2_boxes, 4)
    b2 = K.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    # (..., num_b2_boxes, 2)
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    # (..., num_b2_boxes, 1)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    # (..., 1)
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    # (1, num_b2_boxes)
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    # b1_area + b2_area 的 shape 为 (..., num_b2_boxes)
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


def yolo_loss(args, anchors, num_classes, ignore_thresh=.5, print_loss=True):
    """
    Return yolo_loss tensor

    Args:
        args (list): args[:num_output_layers] the output of yolo_body or tiny_yolo_body
            args[num_output_layers:] y_true
        anchors (np.array): shape=(N, 2), wh
        num_classes (int):
        ignore_thresh (float): the iou threshold whether to ignore object confidence loss
        print_loss:

    Returns:
        loss: tensor, shape=(1,)

    """
    num_output_layers = len(anchors) // 3
    yolo_outputs = args[:num_output_layers]
    y_true = args[num_output_layers:]
    anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]] if num_output_layers == 3 else [[3, 4, 5], [1, 2, 3]]
    # K.shape(yolo_output[0])[1:3] 为 (image_height // 32, image_width // 32), 再乘以 32 就得到原图的大小了
    input_shape = K.cast(K.shape(yolo_outputs[0])[1:3] * 32, K.dtype(y_true[0]))
    # 三个 feature_map 的大小
    grid_shapes = [K.cast(K.shape(yolo_outputs[l])[1:3], K.dtype(y_true[0])) for l in range(num_output_layers)]
    loss = 0
    # batch size
    batch_size = K.shape(yolo_outputs[0])[0]
    batch_size_f = K.cast(batch_size, K.dtype(yolo_outputs[0]))

    for l in range(num_output_layers):
        # 第 4 位为表示是否是 object
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        object_mask = y_true[l][..., 4:5]
        # 第 5 位及其之后的表示属于每个 class 的 prob
        true_class_probs = y_true[l][..., 5:]
        # grid 表示 feature map 上各个 cell 的坐标
        # raw_pred 表示 feature_map 上的 prediction
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, num_classes + 5)
        # pred_xy 为根据 prediction 计算出预测 box 的中心点坐标, normalize 后的
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 2)
        # pred_wh 为根据 prediction 计算出预测 box 的宽和高, normalize 后的
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 2)
        grid, raw_pred, pred_xy, pred_wh = yolo_head(yolo_outputs[l], anchors[anchor_mask[l]], num_classes,
                                                     input_shape, calc_loss=True)
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 4)
        pred_box = K.concatenate([pred_xy, pred_wh])
        # Darknet raw box to calculate loss.
        # y_true[l][..., :2] 是 normalized 过的 center_x, center_y, gt_size/input_size 的 结果
        # raw_true_xy 表示的是中心点的 delta_x delta_y, (batch_size, f_h, f_w, num_anchors_this_layer, 2)
        raw_true_xy = y_true[l][..., :2] * grid_shapes[l][::-1] - grid
        # y_true[l][..., 2:4] 是 normalized 过的 gt_box 的 width 和 height, 乘以 input_shape[::-1] 获得原来 gt_box 的宽高
        # log(gt_boxes_width_height/anchor_width_height)
        raw_true_wh = K.log(y_true[l][..., 2:4] / anchors[anchor_mask[l]] * input_shape[::-1])
        # avoid log(0)=-inf
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh))
        box_loss_scale = 2 - y_true[l][..., 2:3] * y_true[l][..., 3:4]

        # Find ignore mask, iterate over each of batch.
        # 动态 tensor 数组, 参考 https://blog.csdn.net/guolindonggld/article/details/79256018
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')

        def loop_body(b, ignore_mask_):
            # tf.boolean_mask 参考 https://blog.csdn.net/cuicheng01/article/details/80190547
            # 第一个参数的 shape 为 (grid_height, grid_width, num_anchors_this_layer, 4)
            # 第二个参数的 shape 为 (grid_height, grid_width, num_anchors_this_layer)
            # 返回值相当于把第二个参数为 true 的对应的第一个参数的值放到一个 list 里面, 再转换成 np.array
            # (num_gt_boxes, 4)
            gt_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
            # (batch_size, grid_height, grid_width, num_anchors_this_layer, num_gt_boxes)
            iou = box_iou(pred_box[b], gt_box)
            # (batch_size, grid_height, grid_width, num_anchors_this_layer)
            best_iou = K.max(iou, axis=-1)
            ignore_mask_ = ignore_mask_.write(b, K.cast(best_iou < ignore_thresh, K.dtype(gt_box)))
            return b + 1, ignore_mask_

        # 第一个参数是 loop 的条件, 第二个参数是 loop 的内容, 第三个参数作为第一次条件判断和 loop_body 的参数值
        # 以后每一次 loop_body 返回值作为新的用于条件判断的值和 loop_body 的参数值
        _, ignore_mask = K.control_flow_ops.while_loop(lambda b, *largs: b < batch_size, loop_body, [0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # (batch_size, grid_height, grid_width, num_anchors_this_layer, 1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        # K.binary_crossentropy is helpful to avoid exp overflow.
        xy_loss = object_mask * box_loss_scale * K.binary_crossentropy(raw_true_xy, raw_pred[..., 0:2],
                                                                       from_logits=True)
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh - raw_pred[..., 2:4])
        # 1 - object_mask 应该就是论文中提到的 not the best
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                              from_logits=True) + \
                          (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                    from_logits=True) * ignore_mask
        class_loss = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[..., 5:], from_logits=True)
        xy_loss = K.sum(xy_loss) / batch_size_f
        wh_loss = K.sum(wh_loss) / batch_size_f
        confidence_loss = K.sum(confidence_loss) / batch_size_f
        class_loss = K.sum(class_loss) / batch_size_f
        loss += xy_loss + wh_loss + confidence_loss + class_loss
        if print_loss:
            # print_ops = tf.print('\nloss: ', loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask))
            loss = tf.Print(loss, [loss, xy_loss, wh_loss, confidence_loss, class_loss, K.sum(ignore_mask)],
                            message='\nloss: ')
    return loss


if __name__ == '__main__':
    from keras.layers import Input
    image_input = Input(shape=(None, None, 3))
    yolo_body(image_input, 3, 3)
