import cv2
import numpy as np
from math import hypot
from PIL import ImageDraw
from typing import List, Tuple
from PIL import Image as pillow

ndarray, Box, Point = List, Tuple[int, int, int, int], Tuple[int, int]

def overlap(new_box: Tuple, existing_boxes: Tuple)-> bool:
    if not existing_boxes: return False
    x1,y1,w1,h1 = new_box
    for old_box in existing_boxes:
        x2,y2,w2,h2 = old_box
        if(x1 >= (x2+w2) or x2 >= (x1+w1) or y1 >= (y2+h2) or y2 >= (y1+h1)):
            continue
        return True
    return False

def get_centroids(boxes: List[Box])-> List[Box]:
    centroids = []
    for box in boxes:
        x,y,w,h = box
        centroid = (x+(w//2), y+(h//2))
        centroids.append(centroid)
    return centroids

def get_boxes(gray_cv2_image: ndarray, contours: ndarray, minval: int)-> List:
    bgr_cv2_image = cv2.cvtColor(gray_cv2_image, cv2.COLOR_GRAY2BGR)
    boxes = []
    for i, contour in enumerate(contours):
        if not minval < cv2.contourArea(contour) < 100000: continue
        box = cv2.boundingRect(contour)
        x,y,w,h = box
        #if not overlap(box, boxes):
        boxes.append(box)
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

def get_contours(image: ndarray, make_boxes: bool=True)-> ndarray:
    bgr_cv2_image  = cv2.cvtColor(np.asarray(image), cv2.COLOR_GRAY2BGR)
    gray_cv2_image = cv2.cvtColor(bgr_cv2_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_cv2_image, 2, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if make_boxes:
        return get_boxes(gray_cv2_image, contours, 100)
    return gray_cv2_image, contours
    
def draw_my_boxes(imgPath: ndarray, boxes: set, label: bool)-> None:
    image = pillow.open(imgPath).convert('L')
    artist = ImageDraw.Draw(image)
    for box in boxes:
        x, y, w, h = box
        artist.rectangle((x,y,x+w,y+h), fill=None, outline=(0))
        if label:
            artist.text((x,y-10), "({},{})".format(x, y), fill=100, font=None, anchor=None)
            artist.point((xe+(we//2), ye+(he//2)),fill=0)
    image.show()

def draw_my_centroids(imgPath: ndarray, centroids: List[Point])-> None:
    image = pillow.open(imgPath).convert('L')
    blank = pillow.fromarray(np.full_like(image, 0))
    artist = ImageDraw.Draw(blank)
    for centroid in centroids:
        x, y = centroid
        r = 3
        artist.ellipse([(x-r, y-r), (x+r, y+r)], fill=(255))
    blank.show()
    image.show()
