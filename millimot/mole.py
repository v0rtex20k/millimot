import cv2
import numpy as np
import pytesseract
from scipy import stats
import scipy.ndimage as sn
from PIL import Image as pillow
from matplotlib import pyplot as mtplt
from typing import List, Callable, Tuple

ndarray = List
mo = lambda arr: stats.mode(arr, axis=None)[0]

def most_common_pixel(box: Tuple[int, int, int, int], image_arr: ndarray)-> int:
	x, y, w, h = box
	cropped_arr = image_arr[y:y+h,x:x+w]
	pixels, counts = np.unique(cropped_arr, return_counts=True)
	return pixels[np.argmax(counts)]

def grow_horizontal(start: int, stop: int, const: int, direction: int, image_arr: ndarray)-> int:
	border = np.array(image_arr[start:stop, const])
	p = 1
	if border.size == 0: return p
	while 240 > mo(border):
		try:
			border = np.array(image_arr[start-p:stop+p, const+(direction*p)])
			p += 1
		except IndexError as e:
			break # --> out of bounds
	return p

def grow_vertical(start: int, stop: int, const: int, direction: int, image_arr: ndarray)-> int:
	border = np.array(image_arr[const, start:stop])
	p = 1
	if border.size == 0: return p
	while 240 > mo(border):
		try:
			border = np.array(image_arr[const+(direction*p), start-p:stop+p])
			p += 1
		except IndexError as e:
			break # --> out of bounds
	return p

def expanded_box(box: Tuple[int, int, int, int], src_image: ndarray)-> Tuple[int, int, int, int]:
	x, y, w, h = box
	w -= (x+1) # just to adjust for the fact that w and h ALREADY CONTAIN x and y --->
	h -= (y+1) # out of bounds otherwise
	image_arr = np.asarray(src_image).copy()
	left_pad = grow_horizontal(y, y+h, x, -1, image_arr)
	right_pad = grow_horizontal(y, y+h, x+w, 1, image_arr)
	up_pad = grow_vertical(x, x+w, y, -1, image_arr)
	down_pad = grow_vertical(x, x+w, y+h, 1, image_arr)

	return left_pad+1, right_pad+1, up_pad+1, down_pad+1

def trimmed_box(box: Tuple[int, int, int, int], src_image: ndarray)-> Tuple[int, int, int, int]:
	x, y, w, h = box
	image_arr = np.asarray(src_image).copy()
	from_left_cut = 0
	for col in range(w//4):
		strip = image_arr[y:y+h, x+col]
		if strip.size == 0:
			break
		if mo(strip) >= 225:
			from_left_cut += 1
	from_right_cut = 0
	for col in range(w-1, 3*w//4, -1):
		strip = image_arr[y:y+h, x+col]
		if strip.size == 0:
			break
		if mo(strip) >= 225:
			from_right_cut += 1

	return (x+from_left_cut, y, w-(from_left_cut+from_right_cut), h)

area = lambda x, y, w, h: (w-x) * (h-y)
def filter_by_area(nodes: List[Tuple[int, int, int, int]])-> ndarray:
	node_areas = [area(*node) for node in nodes]
	mu = np.mean(node_areas, axis=0)
	stdev = np.std(node_areas, axis=0)
	final_indices = [i for i, a in enumerate(node_areas) if (mu - stdev < a < mu + 3*stdev)]
	return np.asarray(nodes)[final_indices]

def contains_node(box: Tuple[int, int, int, int], src_image: ndarray)-> bool:
	x, y, w, h = box
	image_arr = np.asarray(src_image).copy()
	z = image_arr[y:y+h,x:x+w]
	if h > w or most_common_pixel(box, image_arr) >= 250:
		return False
	return True

