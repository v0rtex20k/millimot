import cv2
import numpy as np
from PIL import Image as pillow
from typing import Any, List, Tuple, Callable

ndarray = List
Box = Tuple[int, int, int, int]
Point = Tuple[int, int]

def blackout(imgPath: ndarray, boxes: List[Box])-> ndarray:
    image = pillow.open(imgPath).convert('L')
    image_arr = np.asarray(image).copy()
    for box in boxes:
        x,y,w,h = box
        image_arr[y:y+h,x:x+w] = 100
    return pillow.fromarray(np.uint8(image_arr))

def overlap(new_box: Tuple, existing_boxes: Tuple)-> bool:
    if not existing_boxes: return False
    x1,y1,w1,h1 = new_box
    for old_box in existing_boxes:
        x2,y2,w2,h2 = old_box
        if(x1 >= (x2+w2) or x2 >= (x1+w1) or y1 >= (y2+h2) or y2 >= (y1+h1)):
            continue
        return True
    return False

def condense(box_groups: List[ndarray[Box]])-> List[Box]:
    condensed_boxes = []
    for group in box_groups:
        x,y,w,h = group.mean(axis=0).astype(int)
        condensed_boxes.append((x,y,w,h))
    return condensed_boxes

def get_centroids(boxes: List[Box])-> List[Box]:
    centroids = []
    centroid_to_box = dict()
    for box in boxes:
        x,y,w,h = box
        centroid = (x+(w//2), y+(h//2))
        centroids.append(centroid)
        centroid_to_box[centroid] = box
    return centroids, centroid_to_box

def get_boxes(gray_cv2_image: ndarray, contours: ndarray, minval: int)-> List:
    boxes = []
    for i, contour in enumerate(contours):
        if not minval < cv2.contourArea(contour) < 100000: continue
        box = cv2.boundingRect(contour)
        boxes.append(box)
    return boxes

def get_contours(image: ndarray, make_boxes: bool=True)-> ndarray:
    bgr_cv2_image  = cv2.cvtColor(np.asarray(image), cv2.COLOR_GRAY2BGR)
    gray_cv2_image = cv2.cvtColor(bgr_cv2_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray_cv2_image, 2, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if make_boxes:
        return gray_cv2_image, get_boxes(gray_cv2_image, contours, 100)
    return gray_cv2_image, contours

def rando(cutoff: int, if_true: Any, if_false: Any)-> int:
    r = np.random.randint(100)
    return (if_true if r > cutoff else if_false)

def erode(ablated_image: ndarray)-> ndarray:
    bgr_cv2_image  = cv2.cvtColor(np.asarray(ablated_image), cv2.COLOR_GRAY2BGR)
    gray_cv2_image = cv2.cvtColor(bgr_cv2_image, cv2.COLOR_BGR2GRAY)
    vd_kernel = np.uint8([[1,0,0,1,0,0,1], # vertical & diagonal
                          [0,1,0,1,0,1,0],
                          [0,0,1,1,1,0,0],
                          [0,0,0,1,0,0,0],
                          [0,0,1,1,1,0,0],
                          [0,1,0,1,0,1,0],
                          [1,0,0,1,0,0,1]])
    eroded_gray_cv2_image = cv2.erode(gray_cv2_image, vd_kernel, iterations=1)
    eroded_arr = np.uint8(eroded_gray_cv2_image)
    eroded_arr[eroded_arr < 100] = 0
    eroded_arr[eroded_arr > 100] = 255
    return pillow.fromarray(eroded_arr)
