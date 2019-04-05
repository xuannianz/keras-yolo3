import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image
import os
import glob
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        # noinspection PyBroadException
        try:
            image = Image.open(img)
        except Exception:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()


def detect_image_dir(yolo):
    while True:
        images_dir = input('Input images dir:')
        # noinspection PyBroadException
        try:
            image_paths = glob.glob(osp.join(images_dir, '*.jpg'))
            for image_path in image_paths:
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                r_image.show()
        except Exception as e:
            print('Open Error! Try again!')


if __name__ == '__main__':
    #############################################################
    # Command line options
    #############################################################

    # class YOLO defines the default value, so suppress any default here
    # argument_default=argparse.SUPPRESS 所有参数如果命令行没有指定, 那么就没有该属性
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        '--model', type=str, dest='model_path',
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str, dest='anchors_path',
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str, dest='classes_path',
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )
    # Command line optional arguments -- for video detection mode
    parser.add_argument(
        "--input", nargs='?', type=str, required=False, default='./path2your_video',
        help="Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        print("Image detection mode")
        if "input" in FLAGS:
            print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
        # vars 获取对象的所有属性和属性值, 返回值是一个 dict
        detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
