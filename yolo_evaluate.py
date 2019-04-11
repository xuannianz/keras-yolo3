from yolo import YOLO
from timeit import default_timer as timer
from yolo3.utils import resize_image
from PIL import Image
import numpy as np
import keras.backend as K
import glob
import os
import os.path as osp
from tqdm import tqdm, trange
import cv2
import h5py
import sys
import pickle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def intersection_area(boxes1, boxes2, mode='outer_product', border_pixels='half'):
    """
    Computes the intersection areas of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively. They must be in corner format.

    In 'outer_product' mode, returns an `(m,n)` matrix with the intersection areas for all possible combinations of the
    boxes in `boxes1` and `boxes2`.
    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation of the `mode` argument
    for details.

    Arguments:
        boxes1 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            corner format or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            corner format or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'.
            In 'outer_product' mode, returns an `(m,n)` matrix with the intersection areas for all possible combinations
            of the `m` boxes in `boxes1` with the `n` boxes in `boxes2`.
            In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2` must be
            broadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of length
            `m` where the i-th position contains the intersection area of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'. If 'include', the border pixels belong
            to the boxes. If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong
            to the boxes, but not the other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values with
        the intersection areas of the boxes in `boxes1` and `boxes2`.
    """

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("All boxes must consist of 4 coordinates, "
                         "but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively."
                         .format(boxes1.shape[1], boxes2.shape[1]))
    if mode not in {'outer_product', 'element-wise'}:
        raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.", format(mode))

    # The number of boxes in `boxes1`
    m = boxes1.shape[0]
    # The number of boxes in `boxes2`
    n = boxes2.shape[0]

    if border_pixels == 'half':
        d = 0
    # If border pixels are supposed to belong to the bounding boxes,
    # we have to add one pixel to any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'include':
        d = 1
    # If border pixels are not supposed to belong to the bounding boxes,
    # we have to subtract one pixel from any difference `xmax - xmin` or `ymax - ymin`.
    elif border_pixels == 'exclude':
        d = -1
    else:
        raise ValueError('`border_pixels` must be one of half, include and exclude')

    # Compute the intersection areas.
    if mode == 'outer_product':
        # For all possible box combinations, get the greater 0 and 1 values.
        # This is a tensor of shape (m,n,2).
        # np.expand_dims 先把 boxes 变成三维的, boxes1 变成 (m, 1, 2), boxes2 变成 (1, n, 2)
        # np.tile 把 boxes1 和 boxes2 变成相同的 shape (m, n, 2)
        # np.maximum 进行 element-wise 的比较
        min_xy = np.maximum(np.tile(np.expand_dims(boxes1[:, [0, 1]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [0, 1]], axis=0), reps=(m, 1, 1)))

        # For all possible box combinations, get the smaller 2 and 3 values.
        # This is a tensor of shape (m,n,2).
        max_xy = np.minimum(np.tile(np.expand_dims(boxes1[:, [2, 3]], axis=1), reps=(1, n, 1)),
                            np.tile(np.expand_dims(boxes2[:, [2, 3]], axis=0), reps=(m, 1, 1)))

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)
        # side_lengths[:, :, 0] 表示 width, side_lengths[:, :, 1] 表示 height, 相乘就表示 area
        return side_lengths[:, :, 0] * side_lengths[:, :, 1]

    elif mode == 'element-wise':
        # 假设此时 boxes1[:, [0, 1]] shape 是 (1, 2), boxes2[:, [0, 1]] 的 shape 是 (n, 2),
        # 在做 maximum, minimum 操作时, 先把 boxes1 广播, 变成 (n, 2) 在逐个比较每个位置上的元素, 返回结果的 shape 也是 (n, 2)
        min_xy = np.maximum(boxes1[:, [0, 1]], boxes2[:, [0, 1]])
        max_xy = np.minimum(boxes1[:, [2, 3]], boxes2[:, [2, 3]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy + d)
        return side_lengths[:, 0] * side_lengths[:, 1]


def iou(boxes1, boxes2, mode='outer_product', border_pixels='half'):
    """
    Computes the intersection-over-union similarity (also known as Jaccard similarity) of two sets of axis-aligned 2D
    rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    In 'outer_product' mode, returns an `(m,n)` matrix with the IoUs for all possible combinations of the boxes in
        `boxes1` and `boxes2`.
    In 'element-wise' mode, `m` and `n` must be broadcast-compatible. Refer to the explanation of the `mode` argument
        for details.

    Arguments:
        boxes1 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            corner format or a 2D Numpy array of shape `(m, 4)` containing the coordinates for `m` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes2`.
        boxes2 (np.array): Either a 1D Numpy array of shape `(4, )` containing the coordinates for one box in the
            corner format or a 2D Numpy array of shape `(n, 4)` containing the coordinates for `n` boxes.
            If `mode` is set to 'element_wise', the shape must be broadcast-compatible with `boxes1`.
        mode (str, optional): Can be one of 'outer_product' and 'element-wise'.
            In 'outer_product' mode, returns an `(m,n)` matrix with the IoU overlaps for all possible combinations of
            the `m` boxes in `boxes1` with the `n` boxes in `boxes2`.
            In 'element-wise' mode, returns a 1D array and the shapes of `boxes1` and `boxes2` must be
            broadcast-compatible. If both `boxes1` and `boxes2` have `m` boxes, then this returns an array of
            length `m` where the i-th position contains the IoU overlap of `boxes1[i]` with `boxes2[i]`.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes.
            Can be 'include', 'exclude', or 'half'.
            If 'include', the border pixels belong to the boxes.
            If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong to the boxes, but not the
            other.

    Returns:
        A 1D or 2D Numpy array (refer to the `mode` argument for details) of dtype float containing values in [0,1],
        the Jaccard similarity of the boxes in `boxes1` and `boxes2`.
        0 means there is no overlap between two given boxes,
        1 means their coordinates are identical.
    """

    #########################################################################################
    # Check for arguments' validation
    #########################################################################################
    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2:
        raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2:
        raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1:
        boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1:
        boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 4):
        raise ValueError("All boxes must consist of 4 coordinates, "
                         "but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
            boxes1.shape[1], boxes2.shape[1]))
    if mode not in {'outer_product', 'element-wise'}:
        raise ValueError("`mode` must be one of 'outer_product' and 'element-wise', but got '{}'.".format(mode))

    #########################################################################################
    # Compute the IoU
    #########################################################################################

    # Compute the intersection areas.
    intersection_areas = intersection_area(boxes1, boxes2, mode=mode)

    # The number of boxes in `boxes1`
    m = boxes1.shape[0]
    # The number of boxes in `boxes2`
    n = boxes2.shape[0]

    # Compute the union areas.
    # Set the correct coordinate indices for the respective formats.
    if border_pixels == 'half':
        d = 0
    # If border pixels are supposed to belong to the bounding boxes,
    # we have to add one pixel to any difference `x_max - x_min` or `y_max - y_min`.
    elif border_pixels == 'include':
        d = 1
    # If border pixels are not supposed to belong to the bounding boxes,
    # we have to subtract one pixel from any difference `x_max - x_min` or `y_max - y_min`.
    elif border_pixels == 'exclude':
        d = -1
    else:
        raise ValueError('`border_pixels` must be one of half, include and exclude')

    if mode == 'outer_product':
        # 每一行 n 个相同的数, 表示 boxes1 中某个 box 的 area, 一共 m 行
        boxes1_areas = np.tile(
            np.expand_dims((boxes1[:, 2] - boxes1[:, 0] + d) * (boxes1[:, 3] - boxes1[:, 1] + d), axis=1),
            reps=(1, n))
        # 每一行 n 个不同的数, 表示 boxes2 中所有 boxes 的 area, m 行都相同
        boxes2_areas = np.tile(
            np.expand_dims((boxes2[:, 2] - boxes2[:, 0] + d) * (boxes2[:, 3] - boxes2[:, 1] + d), axis=0),
            reps=(m, 1))
    # mode == 'element-wise'
    else:
        # 假设 boxes1 的 shape 为 (1, 4) 那么 boxes1_areas 的 shape 是 (1,)
        # boxes2 的 shape 为 (n, 4), 那么 boxes2_areas 的 shape 就是 (n, )
        # 后面两者相加时做一次广播, 相加结果的 shape 也是 (n, )
        boxes1_areas = (boxes1[:, 2] - boxes1[:, 0] + d) * (boxes1[:, 3] - boxes1[:, 1] + d)
        boxes2_areas = (boxes2[:, 2] - boxes2[:, 0] + d) * (boxes2[:, 3] - boxes2[:, 1] + d)

    # boxes1_areas + boxes2_area 的 shape 为 (m,n), (m,) or (n,)
    # 如果是 (m,n), 每一行表示 boxes1 中某个 box 的 area 和 boxes2 中所有 box 的 area 的和
    # 如果是 (m,), 每一个元素表示 boxes1 中某个 box 的 area 和 boxes2 中 box 的 area 的和
    # 如果是 (n,), 每一个元素表示 boxes2 中某个 box 的 area 和 boxes1 中 box 的 area 的和
    union_areas = boxes1_areas + boxes2_areas - intersection_areas

    return intersection_areas / union_areas


def get_predictions_per_class(yolo,
                              model_input_size,
                              image_paths,
                              num_classes=20,
                              ):
    """

    Args:
        yolo: yolo model
        model_input_size (tuple): (w, h)
        image_paths (tuple/list):
        num_classes (int):

    Returns:

    """
    num_images = len(image_paths)
    batch_size = 16
    # 第 0 个元素表示 background
    predictions_per_class = [list() for _ in range(num_classes + 1)]

    for i in range(0, num_images, batch_size):
        if i + batch_size > num_images:
            batch_image_paths = image_paths[i:]
        else:
            batch_image_paths = image_paths[i:i + batch_size]
        batch_images_data = []
        for image_path in batch_image_paths:
            image = Image.open(image_path)
            boxed_image = resize_image(image, model_input_size)
            image_data = np.array(boxed_image, dtype='float32')
            # (h, w, 3)
            # print(image_data.shape)
            image_data /= 255.
            batch_images_data.append(image_data)
        batch_images_data = np.array(batch_images_data)
        batch_pred_boxes, batch_pred_scores, batch_pred_classes = yolo.sess.run(
            [yolo.boxes, yolo.scores, yolo.classes],
            feed_dict={
                yolo.yolo_model.input: batch_images_data,
                yolo.image_shape: np.stack([[416, 416]] * len(batch_image_paths)),
                K.learning_phase(): 0
            })
        for j, batch_item_pred_boxes in enumerate(batch_pred_boxes):
            # 把填充的部分去掉
            batch_item_pred_classes = batch_pred_classes[j]
            batch_item_pred_scores = batch_pred_scores[j]
            batch_item_pred_boxes = batch_item_pred_boxes[batch_item_pred_scores > 0.0]
            batch_item_pred_classes = batch_item_pred_classes[batch_item_pred_scores > 0.0]
            batch_item_pred_scores = batch_item_pred_scores[batch_item_pred_scores > 0.0]
            image_id = osp.split(batch_image_paths[j])[-1][:-4]
            # print('Found {} boxes for image {}'.format(len(pred_boxes), image_id))
            for k, pred_box in enumerate(batch_item_pred_boxes):
                class_id = int(batch_item_pred_classes[k])
                score = batch_item_pred_scores[k]
                ymin = int(round(pred_box[0]))
                xmin = int(round(pred_box[1]))
                ymax = int(round(pred_box[2]))
                xmax = int(round(pred_box[3]))
                prediction = (image_id, score, xmin, ymin, xmax, ymax)
                # 第 0 个元素表示 background
                predictions_per_class[int(class_id) + 1].append(prediction)
    return predictions_per_class


def get_hdf5_data(hdf5_dataset_path,
                  num_classes=20,
                  verbose=True,
                  ret=True):
    """
    Counts the number of ground truth boxes for each class across the dataset.

    获取 self.data_generator.labels, 每一个元素是一个 np.array, 表示一个 image 上的所有 gt_boxes 的坐标和 class_id
    遍历这些 np.array, 根据 gt_boxes 的 class_id 统计每一个 class 的 gt_boxes 的数量
    返回一个 len=num_classes+1 的数组, 每一个元素表示每一个 class 的 gt_boxes 的数量

    Arguments:
        hdf5_dataset_path (str):
        num_classes (int):
        verbose (bool, optional): If `True`, will print out the progress during runtime.
        ret (bool, optional): If `True`, returns the list of counts.

    Returns:
        None by default. Optionally, a list containing a count of the number of ground truth boxes for each class
        across the entire dataset.
    """
    hdf5_dataset = h5py.File(hdf5_dataset_path, 'r')
    labels = []
    labels_dataset = hdf5_dataset['labels']
    label_shapes_dataset = hdf5_dataset['label_shapes']
    image_ids = []
    image_ids_dataset = hdf5_dataset['image_ids']
    dataset_size = len(labels_dataset)
    if verbose:
        tr = trange(dataset_size, desc='Loading labels', file=sys.stdout)
    else:
        tr = range(dataset_size)
    for i in tr:
        labels.append(labels_dataset[i].reshape(label_shapes_dataset[i]))
        image_ids.append(image_ids_dataset[i])

    # 用于表示每个 class 有多少个 gt_boxes, 一个元素表示 background
    num_gt_per_class = np.zeros(shape=num_classes + 1, dtype=np.int)

    if verbose:
        tr = trange(len(labels), file=sys.stdout)
        tr.set_description('Computing the number of positive ground truth boxes per class.')
    else:
        tr = range(len(labels))

    # Iterate over the ground truth for all images in the dataset.
    for i in tr:
        boxes = labels[i]
        # Iterate over all ground truth boxes for the current image.
        for j in range(boxes.shape[0]):
            class_id = boxes[j, 0]
            num_gt_per_class[class_id] += 1

    if verbose:
        print('dataset_size={}'.format(dataset_size))
        print('num_gt_per_class={}'.format(num_gt_per_class))

    if ret:
        return dataset_size, labels, image_ids, num_gt_per_class


def match_predictions(predictions_per_class,
                      dataset_size,
                      labels,
                      image_ids,
                      num_gt_per_class,
                      num_classes=20,
                      matching_iou_threshold=0.5,
                      border_pixels='half',
                      sorting_algorithm='quicksort',
                      verbose=True,
                      ret=True):
    """
    Matches predictions to ground truth boxes.

    Note that `get_predictions_per_class()` must be called before calling this method.
    1. 遍历所有 class 的 predictions
        2. 把 predictions 按 confidence 排序
        3. 按 confidence 从大到小的顺序遍历所有的 predictions
            4. 根据该 prediction 的 image_id, 到  data_generator.labels 找到该 image_id 下对应 class_id 的所有 gt_boxes
            5. 如果 gt_boxes 为空, 该 prediction 为 false positive
            6. 否则计算 prediction 和 gt_boxes 的 iou
            7. 找到最大的 iou 和对应的 gt_box_index
            8. 如果 iou > threshold
                9. 判断这个 gt_box 是否已经被 match 过
                    10. 如果没有, prediction --> true positive
                    11. 如果有, prediction --> false positive
            12 否则 prediction --> false positive
    Arguments:
        predictions_per_class (list):
        dataset_size (int):
        labels (list):
        image_ids (list):
        num_gt_per_class (np.array):
        num_classes (int):
        matching_iou_threshold (float, optional): A prediction will be considered a true positive if it has a
            Jaccard overlap of at least `matching_iou_threshold` with any ground truth bounding box of the same
            class.
        border_pixels (str, optional): How to treat the border pixels of the bounding boxes. Can be 'include',
            'exclude', or 'half'.
            If 'include', the border pixels belong to the boxes.
            If 'exclude', the border pixels do not belong to the boxes.
            If 'half', then one of each of the two horizontal and vertical borders belong to the boxex, but not the
            other.
        sorting_algorithm (str, optional): Which sorting algorithm the matching algorithm should use. This argument
            accepts any valid sorting algorithm for Numpy's `argsort()` function. You will usually want to choose
            between 'quicksort' (fastest and most memory efficient, but not stable) and 'mergesort' (slight slower
            and less memory efficient, but stable). The official Matlab evaluation algorithm uses a stable sorting
            algorithm, so this algorithm is only guaranteed to behave identically if you choose 'mergesort' as the
            sorting algorithm, but it will almost always behave identically even if you choose 'quicksort'
            (but no guarantees).
        verbose (bool, optional): If `True`, will print out the progress during runtime.
        ret (bool, optional): If `True`, returns the true and false positives.

    Returns:
        None by default. Optionally, four nested lists containing the true positives, false positives, cumulative
        true positives, and cumulative false positives for each class.
    """

    if labels is None:
        raise ValueError("Matching predictions to ground truth boxes not possible, no ground truth given.")

    if num_gt_per_class is None:
        raise ValueError("There are no ground truth numbers"
                         "You must run `get_num_gt_per_class()` before calling this method.")
    if predictions_per_class is None:
        raise ValueError("There are no prediction results. "
                         "You must run `get_predictions_per_class()` before calling this method.")

    # Convert the ground truth to a more efficient format for what we need to do, which is access ground truth by
    # image ID repeatedly.
    ground_truth = {}
    # Whether or not we have annotations to decide whether ground truth boxes should be neutral or not.
    for i in range(dataset_size):
        image_id = image_ids[i]
        label = labels[i]
        ground_truth[image_id] = label

    # The true positives for each class, sorted by descending confidence. 第一个 [] 用于表示 background.
    # 其余元素的长度和该元素对应的 predictions 的长度一样
    true_positives = [[]]
    # The false positives for each class, sorted by descending confidence. 第一个 [] 用于表示 background.
    false_positives = [[]]
    cumulative_true_positives = [[]]
    cumulative_false_positives = [[]]

    # Iterate over all non-background classes.
    for class_id in range(1, num_classes + 1):
        predictions = predictions_per_class[class_id]
        # Store the matching results in these lists:
        # 1 for every prediction that is a true positive, 0 otherwise
        true_pos = np.zeros(len(predictions), dtype=np.int)
        # 1 for every prediction that is a false positive, 0 otherwise
        false_pos = np.zeros(len(predictions), dtype=np.int)

        # In case there are no predictions at all for this class, we're done here.
        if len(predictions) == 0:
            print("No predictions for class {}/{}".format(class_id, num_classes))
            true_positives.append(true_pos)
            false_positives.append(false_pos)
            # Cumulative sums of the true positives
            cumulative_true_pos = np.cumsum(true_pos)
            # Cumulative sums of the false positives
            cumulative_false_pos = np.cumsum(false_pos)
            cumulative_true_positives.append(cumulative_true_pos)
            cumulative_false_positives.append(cumulative_false_pos)
            continue

        # Convert the predictions list for this class into a structured array so that we can sort it by confidence.
        # Get the number of characters needed to store the image ID strings in the structured array.
        # Keep a few characters buffer in case some image IDs are longer than others.
        num_chars_of_image_id = len(str(predictions[0][0])) + 6
        # Create the data type for the structured array.
        # 参见 https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html
        # U 表示 unicode, U 和 f 后面的数字表示字节数
        preds_data_type = np.dtype([('image_id', 'U{}'.format(num_chars_of_image_id)),
                                    ('confidence', 'f4'),
                                    ('xmin', 'f4'),
                                    ('ymin', 'f4'),
                                    ('xmax', 'f4'),
                                    ('ymax', 'f4')])
        # Create the structured array
        predictions = np.array(predictions, dtype=preds_data_type)
        # Sort the detections by decreasing confidence.
        descending_indices = np.argsort(-predictions['confidence'], kind=sorting_algorithm)
        predictions_sorted = predictions[descending_indices]

        if verbose:
            tr = trange(len(predictions), file=sys.stdout)
            tr.set_description(
                "Matching predictions to ground truth, class {}/{}.".format(class_id, num_classes))
        else:
            tr = range(len(predictions.shape))

        # Keep track of which ground truth boxes were already matched to a detection.
        # key 为 image_id, value 为 len=this_num_gt_boxes 的 np.array 每一个元素表示该 gt_box 是否已经和 prediction match
        gt_matched = {}
        # Iterate over all predictions.
        for i in tr:
            prediction = predictions_sorted[i]
            image_id = prediction['image_id']
            # Convert the structured array element to a regular array.
            pred_box = np.asarray(list(prediction[['xmin', 'ymin', 'xmax', 'ymax']]))

            # Get the relevant ground truth boxes for this prediction,
            # i.e. all ground truth boxes that match the prediction's image ID and class ID.

            # The ground truth could either be a tuple with `(ground_truth_boxes, eval_neutral_boxes)`
            # or only `ground_truth_boxes`.
            # 找出该 prediction 对应的 image 上所有的 gt boxes
            gt = ground_truth[image_id]
            # 从 gt boxes 过滤出属于当前 class_id 的部分
            class_mask = gt[:, 0] == class_id
            gt = gt[class_mask]
            if gt.size == 0:
                # If the image doesn't contain any objects of this class, the prediction becomes a false positive.
                false_pos[i] = 1
                continue

            # Compute the IoU of this prediction with all ground truth boxes of the same class.
            overlaps = iou(boxes1=gt[:, 1:],
                           boxes2=pred_box,
                           mode='element-wise',
                           border_pixels=border_pixels)

            # For each detection, match the ground truth box with the highest overlap.
            # It's possible that the same ground truth box will be matched to multiple detections.
            # 找和当前 prediction 有最大 iou 的 gt_box 和 overlap
            gt_match_index = np.argmax(overlaps)
            gt_match_overlap = overlaps[gt_match_index]

            # 如果最大 overlap 小于 threshold, 认为 prediction 是 false positive
            if gt_match_overlap < matching_iou_threshold:
                # False positive, IoU threshold violated:
                # Those predictions whose matched overlap is below the threshold become false positives.
                false_pos[i] = 1
            else:
                if image_id not in gt_matched:
                    # True positive:
                    # If the matched ground truth box for this prediction hasn't been matched to a different
                    # prediction already, we have a true positive.
                    true_pos[i] = 1
                    gt_matched[image_id] = np.zeros(shape=(gt.shape[0]), dtype=np.bool)
                    gt_matched[image_id][gt_match_index] = True
                # 这个 gt_boxes 还没有 match
                elif not gt_matched[image_id][gt_match_index]:
                    # True positive:
                    # If the matched ground truth box for this prediction hasn't been matched to a different
                    # prediction already, we have a true positive.
                    true_pos[i] = 1
                    gt_matched[image_id][gt_match_index] = True
                else:
                    # False positive, duplicate detection:
                    # If the matched ground truth box for this prediction has already been matched to a
                    # different prediction previously, it is a duplicate detection for an already detected
                    # object, which counts as a false positive.
                    false_pos[i] = 1

        true_positives.append(true_pos)
        false_positives.append(false_pos)
        # Cumulative sums of the true positives
        cumulative_true_pos = np.cumsum(true_pos)
        # Cumulative sums of the false positives
        cumulative_false_pos = np.cumsum(false_pos)
        cumulative_true_positives.append(cumulative_true_pos)
        cumulative_false_positives.append(cumulative_false_pos)

    if ret:
        return true_positives, false_positives, cumulative_true_positives, cumulative_false_positives


def compute_precision_recall(cumulative_true_positives,
                             cumulative_false_positives,
                             num_gt_per_class,
                             num_classes=20,
                             verbose=True,
                             ret=True):
    """
    Computes the precisions and recalls for all classes.

    Note that `match_predictions()` must be called before calling this method.

    根据 self.cumulative_true_positive, self.cumulative_false_positive 计算 cumulative_precision, cumulative_recall
    Arguments:
        cumulative_true_positives (list):
        cumulative_false_positives (list):
        num_gt_per_class (np.array):
        num_classes (int):
        verbose (bool, optional): If `True`, will print out the progress during runtime.
        ret (bool, optional): If `True`, returns the precisions and recalls.

    Returns:
        None by default.
        Optionally, two nested lists containing the cumulative precisions and recalls for each class.
    """

    if (cumulative_true_positives is None) or (cumulative_false_positives is None):
        raise ValueError("True and false positives not available."
                         " You must run `match_predictions()` before you call this method.")

    if num_gt_per_class is None:
        raise ValueError("Number of ground truth boxes per class not available."
                         "You must run `get_num_gt_per_class()` before you call this method.")

    cumulative_precisions = [[]]
    cumulative_recalls = [[]]

    # Iterate over all classes.
    for class_id in range(1, num_classes + 1):
        if verbose:
            print("Computing precisions and recalls, class {}/{}".format(class_id, num_classes))
        tp = cumulative_true_positives[class_id]
        fp = cumulative_false_positives[class_id]
        # 1D array with shape `(num_predictions,)`
        cumulative_precision = np.where(tp + fp > 0, tp / (tp + fp), 0)
        # 1D array with shape `(num_predictions,)`
        # 注意这里的 num_gt_per_class 已经把 neutral 去掉了
        cumulative_recall = tp / num_gt_per_class[class_id]
        cumulative_precisions.append(cumulative_precision)
        cumulative_recalls.append(cumulative_recall)

    if ret:
        return cumulative_precisions, cumulative_recalls


def compute_average_precisions(cumulative_precisions,
                               cumulative_recalls,
                               num_classes=20,
                               mode='sample',
                               num_recall_points=11,
                               verbose=True,
                               ret=True):
    """
    Computes the average precision for each class.

    Can compute the Pascal-VOC-style average precision in both the pre-2010 (k-point sampling) and post-2010
    (integration) algorithm versions.

    Note that `compute_precision_recall()` must be called before calling this method.

    Arguments:
        cumulative_precisions (list):
        cumulative_recalls (list):
        num_classes (int):
        mode (str, optional): Can be either 'sample' or 'integrate'.
            In the case of 'sample', the average precision will be computed according to the Pascal VOC formula that
            was used up until VOC 2009, where the precision will be sampled for `num_recall_points` recall values.
            In the case of 'integrate', the average precision will be computed according to the Pascal VOC formula
            that was used from VOC 2010 onward, where the average precision will be computed by numerically
            integrating over the whole precision-recall curve instead of sampling individual points from it.
            'integrate' mode is basically just the limit case of 'sample' mode as the number of sample points
            increases. For details, see the references below.
        num_recall_points (int, optional): Only relevant if mode is 'sample'. The number of points to sample from
            the precision-recall-curve to compute the average precisions. In other words, this is the number of
            equidistant recall values for which the resulting precision will be computed. 11 points is the value
            used in the official Pascal VOC pre-2010 detection evaluation algorithm.
        verbose (bool, optional): If `True`, will print out the progress during runtime.
        ret (bool, optional): If `True`, returns the average precisions.

    Returns:
        None by default. Optionally, a list containing average precision for each class.

    References:
        http://host.robots.ox.ac.uk/pascal/VOC/voc2012/htmldoc/devkit_doc.html#sec:ap
    """

    if (cumulative_precisions is None) or (cumulative_recalls is None):
        raise ValueError("Precisions and recalls not available. "
                         "You must run `compute_precision_recall()` before you call this method.")

    if mode not in {'sample', 'integrate'}:
        raise ValueError("`mode` can be either 'sample' or 'integrate', but received '{}'".format(mode))

    average_precisions = [0.0]

    # Iterate over all classes.
    for class_id in range(1, num_classes + 1):
        if verbose:
            print("Computing average precision, class {}/{}".format(class_id, num_classes))

        cumulative_precision = cumulative_precisions[class_id]
        cumulative_recall = cumulative_recalls[class_id]
        average_precision = 0.0

        # 参考 https://github.com/rafaelpadilla/Object-Detection-Metrics
        if mode == 'sample':
            for t in np.linspace(start=0, stop=1, num=num_recall_points, endpoint=True):
                # recall 大于 t 的所有 precision
                cum_prec_recall_greater_t = cumulative_precision[cumulative_recall >= t]
                if cum_prec_recall_greater_t.size == 0:
                    precision = 0.0
                else:
                    # 最大的 precision 作为此 recall 点的 precision
                    precision = np.amax(cum_prec_recall_greater_t)
                average_precision += precision
            average_precision /= num_recall_points
        elif mode == 'integrate':
            # We will compute the precision at all unique recall values.
            # unique_recall_indices 每一个元素分别对应 unique_recalls 的每一个元素在 cumulative_recall 中的下标
            unique_recalls, unique_recall_indices, unique_recall_counts = np.unique(cumulative_recall,
                                                                                    return_index=True,
                                                                                    return_counts=True)
            # Store the maximal precision for each recall value and the absolute difference between any two unique
            # recall values in the lists below. The products of these two numbers constitute the rectangular areas
            # whose sum will be our numerical integral.
            maximal_precisions = np.zeros_like(unique_recalls)
            # adam 设置最后一个 recall 的 maximal precision
            maximal_precisions[-1] = np.amax(cumulative_precision[unique_recall_indices[-1]:])
            recall_deltas = np.zeros_like(unique_recalls)
            # adam 设置第一个 recall_delta, 就是一个 recall 到 0 的距离
            recall_deltas[0] = unique_recalls[0]
            # Iterate over all unique recall values in reverse order. This saves a lot of computation:
            # For each unique recall value `r`, we want to get the maximal precision value obtained
            # for any recall value `r* >= r`. Once we know the maximal precision for the last `k` recall
            # values after a given iteration, then in the next iteration, in order compute the maximal
            # precisions for the last `l > k` recall values, we only need to compute the maximal precision
            # for `l - k` recall values and then take the maximum between that and the previously computed
            # maximum instead of computing the maximum over all `l` values.
            # We skip the very last recall value, since the precision after between the last recall value
            # recall 1.0 is defined to be zero.
            # 从倒数第二个 unique_recall 开始计算和倒数第一个 unique_recall 的最大 precision 组成的矩形面积
            for i in range(len(unique_recalls) - 2, -1, -1):
                # 当前 recall 在 cumulative_recall 中的下标
                begin = unique_recall_indices[i]
                # 下一个 recall 在 cumulative_recall 中的下标
                end = unique_recall_indices[i + 1]
                # When computing the maximal precisions, use the maximum of the previous iteration to
                # avoid unnecessary repeated computation over the same precision values.
                # The maximal precisions are the heights of the rectangle areas of our integral under the
                # precision-recall curve.
                # np.amax(cumulative_precision[begin:end]) 得到当前 recall 的最大 precision
                # maximal_precisions[i + 1] 得到之后 recall 的最大 precision
                # 两者的最大值, 作为当前 recall 和下一个 recall 所组矩形的高
                maximal_precisions[i] = np.maximum(np.amax(cumulative_precision[begin:end]),
                                                   maximal_precisions[i + 1])
                # The differences between two adjacent recall values are the widths of our rectangle areas.
                # adam
                recall_deltas[i + 1] = unique_recalls[i + 1] - unique_recalls[i]
            average_precision = np.sum(maximal_precisions * recall_deltas)
        average_precisions.append(average_precision)

    if ret:
        return average_precisions


def compute_mean_average_precision(average_precisions,
                                   ret=True):
    """
    Computes the mean average precision over all classes.

    Note that `compute_average_precisions()` must be called before calling this method.

    Arguments:
        average_precisions (list):
        ret (bool, optional): If `True`, returns the mean average precision.

    Returns:
        A float, the mean average precision, by default. Optionally, None.
    """

    if average_precisions is None:
        raise ValueError("Average precisions not available."
                         "You must run `compute_average_precisions()` before you call this method.")

    # The first element is for the background class, so skip it.
    mean_average_precision = np.average(average_precisions[1:])

    if ret:
        return mean_average_precision


if __name__ == '__main__':
    DATASET_DIR = '/home/adam/.keras/datasets/VOCdevkit'
    train_hdf5_path = osp.join(DATASET_DIR, '07+12_trainval.h5')
    val_hdf5_path = osp.join(DATASET_DIR, '07_test.h5')
    model_input_size_ = (416, 416)
    image_paths_ = glob.glob('/home/adam/.keras/datasets/VOCdevkit/test/VOC2007/JPEGImages/*.jpg')
    yolo_ = YOLO(model_path='logs/2019-04-09/052-20.628-22.252.h5',
                 anchors_path='model_data/voc_anchors.txt',
                 classes_path='model_data/voc_classes.txt',
                 model_image_size=model_input_size_,
                 )
    predictions_pickle_path = 'predictions.pickle'
    if osp.exists(predictions_pickle_path):
        predictions_per_class_ = pickle.load(open(predictions_pickle_path, 'rb'))
    else:
        predictions_per_class_ = get_predictions_per_class(yolo_,
                                                           model_input_size_,
                                                           image_paths_,
                                                           num_classes=20,
                                                           )
        pickle.dump(predictions_per_class_, open(predictions_pickle_path, 'wb'))
    dataset_size_, labels_, image_ids_, num_gt_per_class_ = get_hdf5_data(val_hdf5_path, num_classes=20, verbose=True)
    _, _, cumulative_true_positives_, cumulative_false_positives_ = match_predictions(predictions_per_class_,
                                                                                      dataset_size_,
                                                                                      labels_,
                                                                                      image_ids_,
                                                                                      num_gt_per_class_)
    cumulative_precisions_, cumulative_recalls_ = compute_precision_recall(cumulative_true_positives_,
                                                                           cumulative_false_positives_,
                                                                           num_gt_per_class_,
                                                                           )
    average_precisions_ = compute_average_precisions(cumulative_precisions_, cumulative_recalls_)
    mean_average_precision_ = compute_mean_average_precision(average_precisions_)
    for i_ in range(1, len(average_precisions_)):
        print("{:<14}{:<6}{}".format(yolo_.class_names[i_ - 1], 'AP', round(average_precisions_[i_], 3)))
    print()
    print("{:<14}{:<6}{}".format('', 'mAP', round(mean_average_precision_, 3)))
