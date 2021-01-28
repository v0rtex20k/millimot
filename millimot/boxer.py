import cv2
import random
import numpy as np
from PIL import Image as pillow
from typing import List, Dict, Tuple, Callable

ndarray = List
Box = Tuple[int, int, int, int]
Point = Tuple[int, int]

def fill_boxes(imgPath: ndarray, boxes: List[Box], fill: int)-> ndarray:
    image = pillow.open(imgPath).convert('L')
    image_arr = np.asarray(image).copy()
    for box in boxes:
        x,y,w,h = box
        image_arr[y:y+h,x:x+w] = fill
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
'''
  a  b  c
  d  e  f
  g  h  i
'''
N = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c+1), (r-1, c-1), (r-1, c+1)]
def past_border(x: int, y: int, image_arr: ndarray)-> bool:
    neighbors = np.asarray([image_arr[r,c] for r,c in N(y,x)])
    white_neighbors = neighbors[neighbors > 250]
    if np.count_nonzero(white_neighbors):
        return True
    return False

# 50 % chance of moving
r = lambda pos: pos + random.choice([-1,0,0,0,0,1])
def vertical_search(centroid: Point, direction: int, image_arr: ndarray)-> int:
    x, y = centroid
    dist = 0
    try:
        while image_arr[y+(dist*direction),x] < 253 and dist < 50:
            dist += 1
            if past_border(r(x), y+(dist*direction), image_arr):
                return dist
    except:
        pass
    return dist

def horizontal_search(centroid: Point, direction: int, image_arr: ndarray)-> int:
    x, y = centroid
    dist = 0
    try:
        while image_arr[y,x+(dist*direction)] < 253 and dist < 50:
            dist += 1
            if past_border(x+(dist*direction), r(y), image_arr):
                return dist
    except:
        pass
    return dist

def expand_from_centroid(image: ndarray, condensed_centroids: List[Point])-> List[Box]:
    image_arr = np.asarray(image).copy()
    boxes = []
    for centroid in condensed_centroids:
        x, y = centroid # what if too short?
        up  = vertical_search(centroid, -1, image_arr)
        down = vertical_search(centroid,  1, image_arr)
        left  = horizontal_search(centroid, -1, image_arr)
        right = horizontal_search(centroid,  1, image_arr)
        boxes.append((x-left, y-up, left+right, up+down))
    return boxes

def condense(box_groups: List[ndarray[Box]])-> List[Box]:
    condensed_boxes = []
    for group in box_groups:
        x = np.amin(group[:,0]).astype(int)
        y = np.amin(group[:,1]).astype(int)
        w = np.amax(group[:,2]).astype(int)
        h = np.amax(group[:,3]).astype(int)
        condensed_boxes.append((x,y,w,h))
    return condensed_boxes

def get_centroids(boxes: List[Box])-> Tuple[List[Point], Dict[Point, Box]]:
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

def grayify(image: ndarray)-> None:
    image_arr = np.asarray(image).copy()
    for y in range(image_arr.shape[0]):
        for x in range(image_arr.shape[1]):
            if image_arr[y,x] > 250:
                try:
                    neighbors = np.asarray([image_arr[r,c] for r,c in N(y,x)])
                except:
                    continue
                if any(neighbors[np.logical_and(neighbors > 100, neighbors < 220)]):
                    image_arr[y,x] = 255
    return pillow.fromarray(np.uint8(image_arr))
