import numpy as np
from PIL import Image

with open('text/icdar_2019_art.txt') as f:
    # for annotation_line in f.readlines()[3616:3648]:
    for annotation_line in f.readlines()[3635:3636]:
        line = annotation_line.split()
        print(line[0])
        image = Image.open(line[0])
        image_width, image_height = image.size
        input_width, input_height = 608, 608
        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
        print(np.min(box[:, :4]), np.max(box[:, :4]), box[:, 4])
        scale = min(input_width / image_width, input_height / image_height)
        resized_width = int(image_width * scale)
        resized_height = int(image_height * scale)
        offset_x = (input_width - resized_width) // 2
        offset_y = (input_height - resized_height) // 2
        image = image.resize((resized_width, resized_height), Image.BICUBIC)
        new_image = Image.new('RGB', (input_width, input_height), (128, 128, 128))
        new_image.paste(image, (offset_x, offset_y))
        # image_data = np.array(new_image) / 255.
        image = np.array(new_image)[:, :, ::-1]
        image = image.copy()
        import cv2

        for b in box:
            cv2.rectangle(image, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 1)
        cv2.imshow('image', image)
        cv2.waitKey(0)

# 3727