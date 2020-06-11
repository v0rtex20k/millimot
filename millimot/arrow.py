import numpy as np
from PIL import ImageDraw
import scipy.ndimage as sn
from PIL import Image as pillow
from typing import List, Callable, Tuple

ndarray= List

def fit_line(box: Tuple[int, int, int, int], image_arr: ndarray)-> Tuple[int, int]:
	x, y, w, h = box
	m, n = image_arr.shape
	a,b = 0,0
	cut = 3
	if m > n:
		try:
			a = image_arr[cut,:]
			b = image_arr[-cut,:]
		except IndexError:
			a = image_arr[0,:]
			b = image_arr[0,:]
	elif n >= m:
		try:
			a = image_arr[:,cut]
			b = image_arr[:,-cut]
		except IndexError:
			a = image_arr[:,0]
			b = image_arr[:,0]
	a_intercept = np.argmin(a)
	b_intercept = np.argmin(b)
	return [(x+a_intercept, y), (x+b_intercept, y+h)] if m > n else [(x, y+a_intercept), (x+w, y+b_intercept)]

def trim_h_arrows(box: Tuple[int, int, int, int], src_image_arr: ndarray, threshold: int=120)-> [Tuple[int,int,int,int], ndarray]:
	x, y, w, h = box
	arrow = src_image_arr[y:y+h, x:x+w]
	black_mask = np.any((arrow <= threshold), axis=0)
	left_cut, right_cut = np.argmax(black_mask), np.argmax(black_mask[::-1])
	return ((x+left_cut,y,w-right_cut,h), arrow[:,black_mask]) if arrow[:,black_mask].size else (None, None)

def trim_v_arrows(box: Tuple[int, int, int, int], src_image_arr: ndarray, threshold: int=120)-> [Tuple[int,int,int,int], ndarray]:
	x, y, w, h = box
	arrow = src_image_arr[y:y+h, x:x+w]
	black_mask = np.any((arrow <= threshold), axis=1)
	top_cut, bottom_cut = np.argmax(black_mask), np.argmax(black_mask[::-1])
	return ((x,y+top_cut,w,h-bottom_cut), arrow[black_mask, :]) if arrow[black_mask, :].size else (None, None)
