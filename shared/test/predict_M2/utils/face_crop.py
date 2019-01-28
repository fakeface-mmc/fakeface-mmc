from scipy import misc
import numpy as np
from PIL import Image
import cv2
import dlib
import glob
import os
import shutil
from io import BytesIO

detector = dlib.get_frontal_face_detector()
from ntpath import basename

def hangulFilePathImageRead(filePath):
    stream = open(filePath.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpyArray = np.asarray(bytes, dtype=np.uint8)

    return cv2.imdecode(numpyArray, cv2.IMREAD_UNCHANGED)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    # return a tuple of (x, y, w, h)
    return x, y, w, h

def get_iou(bb1, bb2):
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[2], bb2[2])
    x_right = min(bb1[1], bb2[1])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1[1] - bb1[0]) * (bb1[3] - bb1[2])
    bb2_area = (bb2[1] - bb2[0]) * (bb2[3] - bb2[2])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def crop(filename):
    filename = filename.decode()
    img = hangulFilePathImageRead(filename)
    name = basename(filename)
    height, width, _ = img.shape
    dets, scores, idx = detector.run(img, 1, -1)
    if len(dets)==0:
        return
    faces = list(map(list, zip(dets, scores)))
    list.sort(faces, key=lambda x: x[1], reverse=True)

    det = faces[0][0]
    x, y, w, h = rect_to_bb(det)
    face = np.array([x, y, x + w, y + h])

    center = [(face[0] + face[2]) / 2, (face[1] + face[3]) / 2]
    size = (np.max([face[2] - face[0], face[3] - face[1]]) / 2)

    face[0] = int(center[0] - size * 1.2)
    face[1] = int(center[1] - size * 1.2)
    face[2] = int(center[0] + size * 1.2)
    face[3] = int(center[1] + size * 1.2)
    if face[0] < 0:
        face[0] = 0
    if face[1] < 0:
        face[1] = 0
    if face[2] > width - 1:
        face[2] = width - 1
    if face[3] > height - 1:
        face[3] = height - 1
    scaled = img[face[1]:face[3], face[0]:face[2], :]
    scaled = misc.imresize(scaled, (64, 64), interp='bilinear')
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    result, encimg = cv2.imencode('.jpg', scaled, encode_param)
    res = BytesIO(encimg)
    val = res.getvalue()

    return val, name



if __name__ == '__main__':
    i = 93
    imgs = glob.glob('./../../data/sample/*/*.jpg')
    for img in imgs:
        crop(img)
        i += 1



