"""Mask Application"""
# !/usr/bin/env python
# coding: utf-8

from re import sub
from time import time
from pathlib import Path
import numpy as np
from cv2 import CascadeClassifier, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH
from cv2 import CASCADE_SCALE_IMAGE, destroyAllWindows, FONT_HERSHEY_PLAIN
from cv2 import getTickCount, getTickFrequency, getWindowProperty, imread
from cv2 import imshow, imwrite, LINE_AA, putText, resize, waitKey
from cv2 import WND_PROP_VISIBLE, VideoCapture

def draw_str(dst, string_value, x_postion=20, y_position=20):
    """Draw a string on the detination image"""
    putText(dst, string_value, (x_postion+1, y_position+1), FONT_HERSHEY_PLAIN,
            1.0, (0, 0, 0), thickness=2, lineType=LINE_AA)
    putText(dst, string_value, (x_postion, y_position), FONT_HERSHEY_PLAIN, 1.0,
            (255, 255, 255), lineType=LINE_AA)

def clock():
    """"""
    return getTickCount() / getTickFrequency()

def detect(img, cascade):
    """"""
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4,
                                     minSize=(30, 30), flags=CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def face_mask(image, mask, shape):
    """"""
    for left_bound, y1, x2, y2 in shape:
        left_bound = int(left_bound)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        x_size = int(x2 - left_bound)
        y_size = int(y2 - y1)

        if isinstance(shape, list):
            scaled_mask = mask
        else:
            scaled_mask = resize(mask, (x_size, y_size))

        # image[y1:y1+scaled_mask.shape[0], left_bound:left_bound+scaled_mask.shape[1]] = scaled_mask
        for color in range(3):
            image[y1:y1 + scaled_mask.shape[0], left_bound:left_bound + scaled_mask.shape[1],
                  color] = scaled_mask[:, :, color] * (scaled_mask[:, :, 3]/255.0) + image[y1:y1 + scaled_mask.shape[0],
                                                       left_bound:left_bound+scaled_mask.shape[1],
                                                       color] * (1.0 - scaled_mask[:, :, 3]/255.0)

def display_fps(image, this_time):
    """Draws the FPS rate on an image frame"""
    frame_time = int(1/(clock() - this_time))
    draw_str(image, f"{frame_time} FPS")
    return clock()
def anorm2(input_value):
    """"""
    return (input_value*input_value).sum(-1)

def anorm(input_value):
    """"""
    return np.sqrt(anorm2(input_value))

def create_capture(source=0):
    """"""
    source = str(source).strip()

    # Win32: handle drive letter ('c:', ...)
    source = sub(r'(^|=)([a-zA-Z]):([/\\a-zA-Z0-9])', r'\1?disk\2?\3', source)
    chunks = source.split(':')
    chunks = [sub(r'\?disk([a-zA-Z])\?', r'\1:', s) for s in chunks]

    source = chunks[0]
    try:
        source = int(source)
    except ValueError:
        pass
    params = dict(s.split('=') for s in chunks[1:])

    cap = None
    cap = VideoCapture(source)
    if 'size' in params:
        width, h = map(int, params['size'].split('x'))
        cap.set(CAP_PROP_FRAME_WIDTH, width)
        cap.set(CAP_PROP_FRAME_HEIGHT, h)

    if cap is None or not cap.isOpened():
        print('Warning: unable to open video source: ', source)
    return cap

def main():
    """"""
    #cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    #nested_fn = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    # https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt.xml
    # Try data/haarcascades/haarcascade_eye.xml also
    cascade = CascadeClassifier("haarcascade_frontalface_alt.xml")
    cam = create_capture(0)

    old_rects = []

    s_img = imread("laughing.png", -1)  # GITS_laughingman.svg")
    time_delay = clock()

    window_title = 'Laughing Man Mask'

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
            for left_bound, y1, x2, y2 in rects:
                x_size = x2 - left_bound
                y_size = y2-y1
                rect_pyth = (y_size*y_size)+(x_size*x_size)

            for left_bound, y1, x2, y2 in old_rects:
                x_size = x2-left_bound
                y_size = y2-y1
                old_pyth = (y_size*y_size)+(x_size*x_size)

            if old_pyth > rect_pyth:
                rects = ((old_rects)*2 + (rects)) / 3

            old_rects = rects
        else:
            old_rects = rects

        vis = img.copy()

        face_mask(vis, s_img, rects)
        time_delay = display_fps(vis, time_delay)

        imshow(window_title, vis)

        ch = waitKey(1)
        if getWindowProperty(window_title, WND_PROP_VISIBLE) < 1:
            break

        # Quiting key bindings
        if ch in [27, 1048603]:  # ESC key to abort, close window
            break
        if ch in [32]:  # Space is also a commonly used value
            break

        # Alternatively, pressing S takes a screenshot
        if ch in [115]:
            screenshot_directory = Path.cwd() / "screenshots"

            if not screenshot_directory.exists():
                screenshot_directory.mkdir(parents=True, exist_ok=True)
            unix_value = int(time())
            file_name = str(screenshot_directory / f"laugh_{unix_value}.bmp")
            imwrite(file_name, img)
            print(f"screenshots/laugh_{unix_value}.bmp saved.")

    destroyAllWindows()

if __name__ == '__main__':
    main()
