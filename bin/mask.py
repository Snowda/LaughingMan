""""""
#!/usr/bin/env python

import sys
import getopt
from cv2 import CascadeClassifier, CASCADE_SCALE_IMAGE, COLOR_BGR2GRAY, cvtColor, destroyAllWindows
from cv2 import equalizeHist, imread, imshow, rectangle, resize, waitKey

# local modules
from video import create_capture
from common import clock, draw_str


def detect(img, cascade):
    """"""
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),
                                     flags=CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects

def draw_rects(img, rects, color):
    """"""
    for x1, y1, x2, y2 in rects:
        rectangle(img, (x1, y1), (x2, y2), color, 2)

def face_mask(image, mask, shape):
    """"""
    for x1, y1, x2, y2 in shape:
        x_size = x2 - x1
        y_size = y2 - y1

        if type(shape) != list:
            scaled_mask = resize(mask, (x_size, y_size))
        else:
            scaled_mask = mask

        # image[y1:y1+scaled_mask.shape[0], x1:x1+scaled_mask.shape[1]] = scaled_mask

        for c in range(3):
            image[y1:y1+scaled_mask.shape[0], x1:x1+scaled_mask.shape[1], c] = scaled_mask[:, :, c] * (scaled_mask[:, :, 3]/255.0) + image[y1:y1+scaled_mask.shape[0], x1:x1+scaled_mask.shape[1], c] * (1.0 - scaled_mask[:,:,3]/255.0)

def display_fps(image, this_time):
    """"""
    frame_time = 1/(clock() - this_time)
    draw_str(image, (20, 20), 'FPS: %.2f' % (frame_time))

    return clock()

def draw_eyes(image, gray, rects, nested):
    for x1, y1, x2, y2 in rects:
        roi = gray[y1:y2, x1:x2]
        vis_roi = image[y1:y2, x1:x2]
        subrects = detect(roi.copy(), nested)
        draw_rects(vis_roi, subrects, (255, 0, 0))

def main():
    """"""
    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    cascade_fn = args.get('--cascade', "../../data/haarcascades/haarcascade_frontalface_alt.xml")
    nested_fn = args.get('--nested-cascade', "../../data/haarcascades/haarcascade_eye.xml")

    cascade = CascadeClassifier(cascade_fn)
    nested = CascadeClassifier(nested_fn)

    cam = create_capture(video_src, fallback='synth:bg=../cpp/lena.jpg:noise=0.05')

    old_rects = []

    s_img = imread("laughing_man.png", -1)  # GITS_laughingman.svg") # laugh.png
    x_offset = y_offset = 0

    time_delay = clock()

    while True:
        _, img = cam.read()
        gray = cvtColor(img, COLOR_BGR2GRAY)
        gray = equalizeHist(gray)

        rects = detect(gray, cascade)

        if type(rects) == list:
            rects = old_rects
        else:
            if type(old_rects) != list:
                for x1, y1, x2, y2 in rects:
                    x_size = x2-x1
                    y_size = y2-y1
                    rect_pyth = (y_size*y_size)+(x_size*x_size)

                for x1, y1, x2, y2 in old_rects:
                    x_size = x2-x1
                    y_size = y2-y1
                    old_pyth = (y_size*y_size)+(x_size*x_size)

                if old_pyth > rect_pyth:
                    rects = ((old_rects)*2 + (rects)) / 3

            old_rects = rects

        vis = img.copy()
        
        face_mask(vis, s_img, rects)
        time_delay = display_fps(vis, time_delay)

        imshow('Laughing Man Mask', vis)

        if 0xFF & waitKey(5) == 27:
            break
    destroyAllWindows()

if __name__ == '__main__':
    main()
