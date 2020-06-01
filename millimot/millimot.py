import cv2
import mole
import arrow
import target
import argparse
import numpy as np
from scipy import stats
from PIL import ImageDraw
import scipy.ndimage as sn
from segmentations import *
from transformations import *
from PIL import Image as pillow
from matplotlib import pyplot as mtplt
from typing import List, Callable, Tuple
from skimage.measure import regionprops

ndarray, filepath = List, str
collection = List[ndarray]

#display_images(OrderedDict({"Original": image, "Enhanced": e_image, "Enhanced + Filtered": ef_image, 
#													  "Enhanced + Filtered + Segmented": sef_image}))
def display_images(imageList: collection)-> None:
	print('Press any key to display next image:\n')
	for name, image in imageList.items():
		image.show()
		c = input('Displaying {} Image.\n'.format(name))
		if 'q' in c.lower(): break
	print('[DONE]')

#plot_pixels(image, ef_image)
def plot_pixels(old_image: ndarray, new_image: ndarray)-> None:
	mtplt.plot(old_image.histogram(), label="Original")
	mtplt.plot(new_image.histogram(), label="EF")
	mtplt.xlabel('Pixel Value')
	mtplt.ylabel('Frequency')
	mtplt.legend(loc="upper left")
	mtplt.yscale('log')
	mtplt.show()

def conditional_call(func: (Callable[[ndarray, int], ndarray]), image: ndarray, args: Tuple, s: bool=False)-> ndarray:
	if s:
		if func is None: return image[1]
		rgb_image, ef_image = image[0], np.asarray(image[1], dtype=np.uint8)
		if func.__name__ == 'watershed_segmenter':
			markers = func(ef_image, *args) if args else func(ef_image)
			markers = cv2.watershed(rgb_image, markers)
			marked_image = np.copy(ef_image)
			marked_image[markers == -1] = 100
			return pillow.fromarray(marked_image)
		try:
			return pillow.fromarray(np.uint8(func(ef_image, *args)[1]) if args else np.uint8(func(ef_image)[1])) if func else ef_image
		except:
			return pillow.fromarray(np.uint8(func(ef_image, *args)) if args else np.uint8(func(ef_image))) if func else ef_image
	return pillow.fromarray(np.uint8(func(image, *args)) if args else np.uint8(func(image))) if func else image

ReLU = lambda x: max(0, x)
def find_nodes(imgPath: filepath, filter_func: (Callable[[ndarray, int], ndarray]), f_args: Tuple,
						   		 enhance_func: (Callable[[ndarray, int], ndarray]), e_args: Tuple,
						  		 segment_func: (Callable[[ndarray, int], ndarray]), s_args: Tuple)-> [List[Tuple[int, int, int, int]], ndarray]:
	rgb_image = cv2.imread(imgPath)
	image = pillow.open(imgPath).convert('L')

	e_image = conditional_call(enhance_func, image, e_args)
	ef_image = conditional_call(filter_func, e_image, f_args)
	sef_image = conditional_call(segment_func, (rgb_image, ef_image), s_args, True)

	gray_cv2_image, contours = target.get_contours(sef_image)
	boxes = target.get_boxes(gray_cv2_image, contours, 200)
	image_arr = np.asarray(image).copy()
	nodes = []

	for box in boxes:
		if mole.contains_node(box, image):
			trim_box = mole.trimmed_box(box, image)
			x, y, w, h = trim_box
			left_pad, right_pad, up_pad, down_pad = mole.expanded_box(trim_box, image)
			xt, yt, wt, ht = ReLU(x-left_pad), ReLU(y-up_pad), ReLU(x+w+right_pad), ReLU(y+h+down_pad)
			image_arr[yt:ht,xt:wt] = 255
			nodes.append((xt, yt, wt, ht)) # wt and ht INCLUDE X AND Y!! Easier for slicing, but NOT default rect format
	#pillow.fromarray(image_arr).show()
	return nodes, image_arr

def find_edges(image_arr: ndarray, binary_threshold: int)-> List[Tuple[int, int, int, int]]:
	segment_func, s_args = segmenters('ags')
	s_image_arr = conditional_call(segment_func, (None, image_arr), s_args, True)
	gray_cv2_image, contours = target.get_contours(s_image_arr)
	image = pillow.fromarray(image_arr)
	artist = ImageDraw.Draw(image)
	edges = []
	for i, contour in enumerate(contours):
		box = cv2.boundingRect(contour)
		x, y, w, h = box
		if 300 < w*h < 10000:
			arrow_box, arrow_arr = arrow.trim_v_arrows(box, image_arr)
			if arrow_arr is None or arrow_box is None: continue
			arrow_box, arrow_arr = arrow.trim_h_arrows(arrow_box, image_arr)
			if arrow_arr is None or arrow_box is None: continue
			xc, yc, wc, hc = arrow_box
			artist.rectangle((xc,yc,xc+wc,yc+hc), fill=None, outline=(0))
			points = arrow.fit_line(arrow_box, arrow_arr)
			artist.line(points, fill =(200), width = 2)
			edges.append(points)
	return edges

def core(args: dict)-> None:
	nodes, arrow_image = find_nodes(args['image'], *filters(args['filter']), *enhancers(args['enhancer']), *segmenters(args['segmenter']))
	edges = find_edges(arrow_image, 127)

if __name__ == '__main__':
	#https://pillow.readthedocs.io/en/stable/reference/Image.html
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image",  help="path to image", type=str)
	parser.add_argument("-f", "--filter", help="index of filter to apply to image", type=str)
	parser.add_argument("-e", "--enhancer", help="index of enhancer to apply to image", type=str)
	parser.add_argument("-s", "--segmenter", help="index of segmenter to apply to image", type=str)
	args = vars(parser.parse_args())
	core(args)

# FILTERS:
	# Mean/Median just kinda blur the image. Need to explore impact on segmentation.
	# Max is interesting - just leaves bubbles and arrow tips - very bright
	# Min really emphasises lines and borders *including* arrows.
	# Sobel/Prewitt are very fuzzy, regardless of axis.
	# Laplacian has spaghetti, but can clearly find edges. 
	# LoG has no spaghetti, clear edges. Contender for min/max?

# min + amu/ags are nice - clear white/black delineation ----> min/ags seems best
# print(pytesseract.image_to_string(e_image))

# artist = ImageDraw.Draw(image)
# artist.rectangle((x,y, x+w,y+h), fill=None, outline=(0))