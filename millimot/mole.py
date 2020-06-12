import cv2
import numpy as np
import pytesseract
from scipy import stats
from PIL import Image as pillow
from matplotlib import pyplot as mtplt
from typing import List, Callable, Tuple

ndarray = List
Box = Tuple[int, int, int, int]
mo = lambda arr: stats.mode(arr, axis=None)[0]

def most_common_pixel(box: Box, image_arr: ndarray)-> int:
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

def expanded_box(box: Box, src_image: ndarray)-> Box:
	x, y, w, h = box
	w -= 1 # need -1 b/c you're indexing, not slicing (end not included in slice, but IS for index)
	h -= 1 
	image_arr = np.asarray(src_image).copy()
	left_pad = grow_horizontal(y, y+h, x, -1, image_arr)
	right_pad = grow_horizontal(y, y+h, x+w, 1, image_arr)
	up_pad = grow_vertical(x, x+w, y, -1, image_arr)
	down_pad = grow_vertical(x, x+w, y+h, 1, image_arr)

	return left_pad+1, right_pad+1, up_pad+1, down_pad+1

def trimmed_box(box: Box, src_image: ndarray)-> Box:
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
def filter_by_area(nodes: List[Box])-> ndarray:
	node_areas = [area(*node) for node in nodes]
	mu = np.mean(node_areas, axis=0)
	stdev = np.std(node_areas, axis=0)
	final_indices = [i for i, a in enumerate(node_areas) if (mu - 1.5*stdev <= a <= mu + 5*stdev)] # DO NOT GO PAST -1.5*stdev !!!!
																							   	   # OK to be more lax on the big side (3-5)
	return [(x,y,w,h) for i, (x,y,w,h) in enumerate(nodes) if i in final_indices]

def contains_node(box: Box, src_image: ndarray)-> bool:
	x, y, w, h = box
	image_arr = np.asarray(src_image).copy()
	z = image_arr[y:y+h,x:x+w]
	if h > w or most_common_pixel(box, image_arr) >= 250: # should be very close to pure white
		return False
	return True

