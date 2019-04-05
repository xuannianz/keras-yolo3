from yolo3.utils import rand
from PIL import Image


def test_get_random_data(jitter=0.3):
    w = 1184
    h = 1920
    new_ar = w / h * rand(1 - jitter, 1 + jitter) / rand(1 - jitter, 1 + jitter)
    scale = rand(.25, 2)
    if new_ar < 1:
        nh = int(scale * h)
        nw = int(nh * new_ar)
    else:
        nw = int(scale * w)
        nh = int(nw / new_ar)
    # image = image.resize((nw, nh), Image.BICUBIC)

    # place image
    dx = int(rand(0, w - nw))
    dy = int(rand(0, h - nh))
    print(new_ar, scale, nw, nh, dx, dy)


for i in range(10):
    test_get_random_data()
