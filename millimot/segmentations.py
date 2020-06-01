import cv2
import numpy as np
import scipy.ndimage as sn
from typing import List, Callable, Tuple

ndarray = List
Segmenter = (Callable[[ndarray, int], ndarray], Tuple)

def watershed_segmenter(image: ndarray, erode_iters: int=2)-> ndarray:
	kernel = np.ones((1,1),np.uint8)
	opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
	blur = cv2.GaussianBlur(opening,(1,1),0)
	ret3,th4 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	sure_bg = cv2.dilate(opening,kernel,iterations=erode_iters)
	dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
	ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
	sure_fg = np.uint8(sure_fg)
	unknown = cv2.subtract(sure_bg,sure_fg)
	ret, markers = cv2.connectedComponents(sure_fg)
	markers = markers+1
	markers[unknown==255] = 0
	markers = markers.astype('int32')
	return markers

def segmenters(key: str)-> Segmenter:
	S = {'nan': (None, None),
		 'wtr': (watershed_segmenter, ()),
		 'toz': (cv2.threshold, (127,255,cv2.THRESH_TOZERO)),
		 'trc': (cv2.threshold, (127, 255, cv2.THRESH_TRUNC)),
		 'bin': (cv2.threshold, (127, 255, cv2.THRESH_BINARY)),
		 'tzi': (cv2.threshold, (127,255,cv2.THRESH_TOZERO_INV)),
		 'biv': (cv2.threshold, (127, 255, cv2.THRESH_BINARY_INV)),
		 'ots': (cv2.threshold, (127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)),
		 'amu': (cv2.adaptiveThreshold, (255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)), # must be bin or biv
		 'ags': (cv2.adaptiveThreshold, (255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))}
	try:
		return(S[key])
	except KeyError:
		print("[SEGMENTER]\t ---> \"{}\" is not a recognized segmenter.".format(key)); exit(1)