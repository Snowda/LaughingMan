"""Mask Application"""
# !/usr/bin/env python
# coding: utf-8

from re import sub
from time import time
from pathlib import Path
from io import BytesIO
import numpy as np
from PIL import Image
import pyvips
from cv2 import CascadeClassifier, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH
from cv2 import CASCADE_SCALE_IMAGE, COLOR_RGBA2BGRA, cvtColor, destroyAllWindows
from cv2 import FONT_HERSHEY_PLAIN, getTickCount, getTickFrequency, getWindowProperty
from cv2 import imread, imshow, imwrite, LINE_AA, putText, resize, waitKey
from cv2 import WND_PROP_VISIBLE, VideoCapture

WINDOW_TITLE = 'Laughing Man Mask'

def draw_str(dst, string_value, x_postion=20, y_position=20):
    """Draw a string on the detination image"""
    putText(dst, string_value, (x_postion+1, y_position+1), FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 0), thickness=2, lineType=LINE_AA)
    putText(dst, string_value, (x_postion, y_position), FONT_HERSHEY_PLAIN, 1.0,
            (255, 255, 255), lineType=LINE_AA)

def clock():
    """"""
    return getTickCount() / getTickFrequency()

def face_mask(image, mask, shape):
    """"""
    for left_bound, lower_bound, right_bound, upper_bound in shape:
        left_bound = int(left_bound)
        lower_bound = int(lower_bound)
        right_bound = int(right_bound)
        upper_bound = int(upper_bound)
        x_size = int(right_bound - left_bound)
        y_size = int(upper_bound - lower_bound)

        if isinstance(shape, list):
            scaled_mask = mask
        else:
            scaled_mask = resize(mask, (x_size, y_size))

        for color in range(3):
            image[lower_bound:lower_bound + scaled_mask.shape[0], left_bound:left_bound + scaled_mask.shape[1],
                  color] = scaled_mask[:, :, color] * (scaled_mask[:, :, 3]/255.0) + image[lower_bound:lower_bound + scaled_mask.shape[0],
                                                       left_bound:left_bound+scaled_mask.shape[1],
                                                       color] * (1.0 - scaled_mask[:, :, 3]/255.0)

def display_fps(image, this_time):
    """Draws the FPS rate on an image frame"""
    frame_time = int(1/(clock() - this_time))
    draw_str(image, f"{frame_time} FPS")
    return clock()

def main():
    """"""
    # Win32: handle drive letter ('c:', ...)
    source = sub(r'(^|=)([a-zA-Z]):([/\\a-zA-Z0-9])', r'\1?disk\2?\3', "0")
    chunks = source.split(':')
    chunks = [sub(r'\?disk([a-zA-Z])\?', r'\1:', s) for s in chunks]

    source = chunks[0]
    try:
        source = int(source)
    except ValueError:
        pass
    params = dict(s.split('=') for s in chunks[1:])

    cam = VideoCapture(source)
    if 'size' in params:
        width, height = map(int, params['size'].split('x'))
        cam.set(CAP_PROP_FRAME_WIDTH, width)
        cam.set(CAP_PROP_FRAME_HEIGHT, height)

    if cam is None or not cam.isOpened():
        print(f"Warning: unable to open {source}")
        return

    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
    # Try data/haarcascades/haarcascade_eye.xml also
    cascade = CascadeClassifier("haarcascade_frontalface_alt.xml")

    old_rects = []

    image = pyvips.Image.new_from_file("laughing-man.svg", dpi=300)

    pil_img = Image.open(BytesIO(image)).convert('RGBA')
    cv_img = cvtColor(np.array(pil_img), COLOR_RGBA2BGRA)
    s_img = imread(cv_img, -1)
    #s_img = imread("laughing-man.svg", -1)
    time_delay = clock()

    while True:
        _, img = cam.read()
        # Face detection
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
                                     minSize=(30, 30), flags=CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            rects = []
        else:
            rects[:, 2:] += rects[:, :2]

        if isinstance(rects, list):
            rects = old_rects
        elif not isinstance(old_rects, list):
            for left_bound, lower_bound, right_bound, upper_bound in rects:
                x_size = right_bound - left_bound
                y_size = upper_bound-lower_bound
                rect_pyth = (y_size*y_size)+(x_size*x_size)

            for left_bound, lower_bound, right_bound, upper_bound in old_rects:
                x_size = right_bound-left_bound
                y_size = upper_bound-lower_bound
                old_pyth = (y_size*y_size)+(x_size*x_size)

            if old_pyth > rect_pyth:
                rects = ((old_rects)*2 + (rects)) / 3

            old_rects = rects
        else:
            old_rects = rects

        vis = img.copy()
        face_mask(vis, s_img, rects)
        time_delay = display_fps(vis, time_delay)
        imshow(WINDOW_TITLE, vis)

        character = waitKey(1)
        if getWindowProperty(WINDOW_TITLE, WND_PROP_VISIBLE) < 1:
            break

        # Quiting key bindings
        if character in [27, 1048603, 32]:  # ESC or Space key to close window
            break

        # Alternatively, pressing S takes a screenshot
        if character in [115]:
            screenshot_directory = Path.cwd() / "screenshots"

            if not screenshot_directory.exists():
                screenshot_directory.mkdir(parents=True, exist_ok=True)
            unix_value = int(time())
            imwrite(str(screenshot_directory / f"laugh_{unix_value}.bmp"), img)
            print(f"screenshots/laugh_{unix_value}.bmp saved.")

    destroyAllWindows()

if __name__ == '__main__':
    main()
