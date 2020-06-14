import cv2
import numpy as np
from scipy import stats
from PIL import ImageDraw
from typing import List, Tuple
from PIL import Image as pillow

ndarray = List
Box = Tuple[int, int, int, int]
Point = Tuple[int, int]

mo = lambda arr: stats.mode(arr, axis=None)[0] # "mo" stands for mode

def draw_my_centroids(centroids: List[Point], constellation: bool=False, imgPath: str=None, src_image: ndarray=None)-> None:
    image = None
    if src_image:
        image = src_image
    else:
        image = pillow.open(imgPath).convert('L')
    artist = None
    if constellation:
        blank = pillow.fromarray(np.full_like(image, 0))
        artist = ImageDraw.Draw(blank)
    else:
        artist = ImageDraw.Draw(image)
    for centroid in centroids:
        x, y = centroid
        r = 3 if constellation else 10
        artist.ellipse([(x-r, y-r), (x+r, y+r)], fill=255)
    if constellation:
        blank.show()
    image.show()

def draw_my_curves(imgPath: ndarray, boxes: set, labels: List)-> None:
    image = pillow.open(imgPath).convert('L')
    image_arr = np.asarray(image)
    artist = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x, y, w, h = box
        cropped = image_arr[y:y+h,x:x+w]
        if mo(cropped) > 250 and np.mean(cropped) > 180:
            artist.rectangle((x,y,x+w,y+h), fill=None, outline=(0))
            if labels:
                artist.text((x,y-10), "({})".format(np.mean(cropped)), fill=100, font=None, anchor=None)
    image.show()

def draw_my_boxes(imgPath: ndarray, boxes: set, labels: List)-> None:
    image = pillow.open(imgPath).convert('L')
    artist = ImageDraw.Draw(image)
    for i, box in enumerate(boxes):
        x, y, w, h = box
        artist.rectangle((x,y,x+w,y+h), fill=None, outline=(0))
        if labels:
            artist.text((x,y-10), "({})".format(labels[i]), fill=100, font=None, anchor=None)
    image.show()

#target.draw_my_contours(gray_cv2_image, contours)
def draw_my_contours(gray_cv2_image: ndarray, contours: ndarray)-> None:
    bgr_cv2_image = cv2.cvtColor(gray_cv2_image, cv2.COLOR_GRAY2BGR)
    for i, contour in enumerate(contours):
        #if not 10 < cv2.contourArea(contour) < 100000: continue
        cv2.drawContours(bgr_cv2_image, contour, -1, (0,255,0), 3)
        #cv2.fillPoly(bgr_cv2_image, pts=contour, color=(0,255,0))
    cv2.imshow('Contours', bgr_cv2_image)
    cv2.waitKey(0)

def plot_pixels(image: ndarray, idx: int)-> None:
    pixels, counts = np.unique(image, return_counts=True)
    mtplt.axvline(pixels[np.argmax(counts)], color='g', linestyle='dashed', linewidth=2)
    mtplt.plot(pixels, counts)
    mtplt.xlabel('Pixel Value')
    mtplt.ylabel('Frequency')
    print('Box {} ---> {:.4f}'.format(str(idx), pixels[np.argmax(counts)]))
    mtplt.title('Box {} ---> {:.4f}'.format(str(idx), image.mean(), pixels[np.argmax(counts)]))
    mtplt.yscale('log')
    mtplt.show()
