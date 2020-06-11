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

def N(r, c, canvas):
	m, n = canvas.shape
	neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c+1), (r-1, c-1), (r-1, c+1)]
	return [(row, col) for row, col in neighbors if (row < m and col < n)]

def burrow(seed: Tuple[int, int], canvas: ndarray, src_image: ndarray)-> ndarray:
	Q = [seed]
	while Q:
		pxl = Q.pop()
		canvas[pxl] = 0
		Q.extend([neighbor for neighbor in N(*pxl, canvas) if (src_image[neighbor] == 0 and canvas[neighbor] != 0)])
	return canvas

def constrain(canvas: ndarray)-> Tuple[int, int, int, int]:
	rows = np.all(canvas, axis=0)
	cols = np.all(canvas, axis=1)
	rows = np.where(rows == True, 0, 1)
	cols = np.where(cols == True, 0, 1)
	rows = np.nonzero(rows)
	cols = np.nonzero(cols)
	return (rows[0][0], cols[0][0], rows[0][-1]-rows[0][0], cols[0][-1]-cols[0][0])

def search_and_destroy(box: Tuple[int, int, int, int], src_image: ndarray)-> ndarray:
	x, y, w, h = box
	canvas = np.full_like(np.asarray(src_image), 255)
	features = []
	for i in range(y, y+h):
		for j in range(x, x+w):
			if src_image[i,j] == 0 and canvas[i,j] != 0:
				feature = constrain(burrow((i,j), canvas, src_image))
				fx, fy, fw, fh = feature
				if not overlap(feature, features):
					features.append(feature)
					#canvas[fy:fy+fh, fx:fx+fw] = 0
					#pillow.fromarray(canvas).show()
	return features

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

def expanded_box(box: Tuple[int, int, int, int], src_image: ndarray, show: bool=False)-> Tuple[int, int, int, int]:
	x, y, w, h = box
	w -= (x+1) # just to adjust for the fact that w and h ALREADY CONTAIN x and y --->
	h -= (y+1) # out of bounds otherwise
	image_arr = np.asarray(src_image).copy()
	left_pad = grow_horizontal(y, y+h, x, -1, image_arr)
	right_pad = grow_horizontal(y, y+h, x+w, 1, image_arr)
	up_pad = grow_vertical(x, x+w, y, -1, image_arr)
	down_pad = grow_vertical(x, x+w, y+h, 1, image_arr)
	if show:
		old_cropped = np.asarray(src_image.crop((x,y,x+w,y+h)))
		pillow.fromarray(old_cropped).show()
		input("\tExpanded ...")
		new_cropped = np.asarray(src_image.crop((x-left_pad,y-up_pad,x+w+right_pad,y+h+down_pad)))
		pillow.fromarray(new_cropped).show()
	return left_pad+1, right_pad+1, up_pad+1, down_pad+1

def trimmed_box(box: Tuple[int, int, int, int], src_image: ndarray, show: bool=False)-> Tuple[int, int, int, int]:
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
	if show:
		old_cropped = np.asarray(src_image.crop((x,y,x+w,y+h)))
		pillow.fromarray(old_cropped).show()
		input("\tTrimmed ...")
		new_cropped = np.asarray(src_image.crop((x+from_left_cut,y,x+w-from_right_cut,y+h)))
		pillow.fromarray(new_cropped).show()
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

