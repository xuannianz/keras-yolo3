"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        # funcs 中有多个函数, 首先前两个元素作为 f, g, 返回 lambda *args, **kwargs: g(f(*args, **kwargs)) 作为新的 f
        # 第三个元素作为新的 g, 依次类推
        return reduce(lambda f, g: lambda *args, **kwargs: g(f(*args, **kwargs)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


def resize_image(image, input_size):
    """
    resize image with unchanged aspect ratio using padding, 把 image resize 成 input_size

    先算出 image.size --> input_size 的较小 scale, 然后按照该 scale resize
    resize 之后 width 或者 height 会不足 input_size, 此时再填充
    Args:
        image:
        input_size: (w, h)

    Returns:

    """
    """"""
    image_width, image_height = image.size
    input_width, input_height = input_size
    scale = min(input_width / image_width, input_height / image_height)
    resized_width = int(image_width * scale)
    resized_height = int(image_height * scale)
    image = image.resize((resized_width, resized_height), Image.BICUBIC)
    new_image = Image.new('RGB', input_size, (128, 128, 128))
    new_image.paste(image, ((input_width - resized_width) // 2, (input_height - resized_height) // 2))
    return new_image


def rand(a=0.0, b=1.0):
    return np.random.rand() * (b - a) + a


def get_random_data(annotation_line, input_shape,
                    random=True,
                    max_boxes=20,
                    jitter=.3,
                    hue=.1,
                    sat=1.5,
                    val=1.5,
                    proc_img=True):
    """
    random preprocessing for real-time data augmentation

    Args:
        annotation_line:
        input_shape:
        random:
        max_boxes:
        jitter:
        hue:
        sat:
        val:
        proc_img:

    Returns:

    """
    line = annotation_line.split()
    image = Image.open(line[0])
    image_width, image_height = image.size
    input_height, input_width = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

    if not random:
        # resize image
        scale = min(input_width / image_width, input_height / image_height)
        resized_width = int(image_width * scale)
        resized_height = int(image_height * scale)
        offset_x = (input_width - resized_width) // 2
        offset_y = (input_height - resized_height) // 2
        image_data = None
        if proc_img:
            image = image.resize((resized_width, resized_height), Image.BICUBIC)
            new_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
            new_image.paste(image, (offset_x, offset_y))
            image_data = np.array(new_image) / 255.

        # correct boxes
        box_data = np.zeros((max_boxes, 5))
        if len(box) > 0:
            np.random.shuffle(box)
            if len(box) > max_boxes:
                box = box[:max_boxes]
            box[:, [0, 2]] = box[:, [0, 2]] * scale + offset_x
            box[:, [1, 3]] = box[:, [1, 3]] * scale + offset_y
            box_data[:len(box)] = box

        return image_data, box_data

    # resize image
    new_ar = input_width / input_height * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * input_height)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * input_width)
        nh = int(nw / new_ar)
    image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, input_width - nw))
    dy = int(rand(0, input_height - nh))
    new_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image = new_image

    # flip image or not
    flip = rand() < .5
    if flip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # distort image
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = rgb_to_hsv(np.array(image) / 255.)
    x[..., 0] += hue
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x > 1] = 1
    x[x < 0] = 0
    # numpy array, 0 to 1
    image_data = hsv_to_rgb(x)

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        np.random.shuffle(box)
        box[:, [0, 2]] = box[:, [0, 2]] * nw / image_width + dx
        box[:, [1, 3]] = box[:, [1, 3]] * nh / image_height + dy
        if flip:
            box[:, [0, 2]] = image_width - box[:, [2, 0]]
        box[:, 0:2][box[:, 0:2] < 0] = 0
        box[:, 2][box[:, 2] > image_width] = image_width
        box[:, 3][box[:, 3] > image_height] = image_height
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box
        if len(box) > max_boxes:
            box = box[:max_boxes]
        box_data[:len(box)] = box

    return image_data, box_data
