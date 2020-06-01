import cv2
import numpy as np
from typing import List, Tuple
import scipy.ndimage as sn
from PIL import Image as pillow

ndarray = List

def overlap(new_box: Tuple, existing_boxes: Tuple)-> bool:
    if not existing_boxes: return False
    x1,y1,w1,h1 = new_box
    for old_box in existing_boxes:
        x2,y2,w2,h2 = old_box
        if(x1 >= (x2+w2) or x2 >= (x1+w1) or y1 >= (y2+h2) or y2 >= (y1+h1)):
            continue
        return True
    return False

def get_boxes(gray_cv2_image: ndarray, contours: ndarray, minval: int, show=False)-> List:
    bgr_cv2_image = cv2.cvtColor(gray_cv2_image, cv2.COLOR_GRAY2BGR)
    boxes = []
    for i, contour in enumerate(contours):
        if not minval < cv2.contourArea(contour) < 100000: continue
        box = cv2.boundingRect(contour)
        x,y,w,h = box
        if not overlap(box, boxes):
            boxes.append(box)
            if show:
                cv2.rectangle(bgr_cv2_image,(x,y),(x+w,y+h),(0,255,0),2)
    if show:
        cv2.imshow('Boxes', bgr_cv2_image)
        cv2.waitKey(0)
    return boxes

#target.draw_my_contours(gray_cv2_image, contours)
def draw_my_contours(gray_cv2_image: ndarray, contours: ndarray)-> None:
    bgr_cv2_image = cv2.cvtColor(gray_cv2_image, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        if not 10 < cv2.contourArea(contour) < 100000: continue
        cv2.drawContours(bgr_cv2_image, contour, -1, (0,255,0), 3)
        cv2.fillPoly(bgr_cv2_image, pts=contour, color=(0,255,0))
    cv2.imshow('Contours', bgr_cv2_image)
    cv2.waitKey(0)

def get_contours(image: ndarray)-> ndarray:
    bgr_cv2_image  = cv2.cvtColor(np.asarray(image), cv2.COLOR_GRAY2BGR)
    gray_cv2_image = cv2.cvtColor(bgr_cv2_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_cv2_image, 2, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return gray_cv2_image, contours
    