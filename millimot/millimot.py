import cv2
import mole
import boxer
import argparse
import advanced
import networker
import graphics
import numpy as np
import networkx as nx
from PIL import ImageDraw
from segmentations import *
from transformations import *
from PIL import Image as pillow
from itertools import permutations, product
from matplotlib import pyplot as mtplt
from typing import List, Callable, Tuple, NewType, Set, Dict

ndarray = List
Point = Tuple[int, int]
collection = List[ndarray]
Box = Tuple[int, int, int, int]
Graph = NewType('Graph', nx.classes.graph.Graph)
Mutator = Callable[[ndarray, int], ndarray]

# SET-UNION MIN, MAX, MEDIAN FILTERS w/ AGS/AMU!!!!! (then run overlap)

def conditional_call(func: Mutator, image: ndarray, args: Tuple, segmenting: bool=False)-> ndarray:
	if func is None: return image
	if not segmenting: return pillow.fromarray(np.uint8(func(image, *args)) if args else np.uint8(func(image))) if func else image
	image = np.asarray(image, dtype=np.uint8)
	try:
		return pillow.fromarray(np.uint8(func(image, *args)[1]) if args else np.uint8(func(image)[1])) if func else image
	except:
		return pillow.fromarray(np.uint8(func(image, *args)) if args else np.uint8(func(image))) if func else image

def mutate(imgPath: str, filter_func: Mutator, f_args: Tuple[int], enhance_func: Mutator, e_args: Tuple[int], 
																   segment_func: Mutator, s_args: Tuple[int])-> ndarray:
	try:
		image = pillow.open(imgPath).convert('L')
	except FileNotFoundError:
		print('No file exists at \"{}\".'.format(imgPath))
		exit()
	clone = image.copy()
	clone = conditional_call(enhance_func, clone, e_args)
	clone = conditional_call(filter_func, clone, f_args)
	clone = conditional_call(segment_func, clone, s_args, segmenting=True)
	return image, clone

def find_node_boxes(image: ndarray, clone: ndarray)-> Set[Box]:
	gray_cv2_image, boxes = boxer.get_contours(clone)
	node_boxes = []
	for box in boxes:
		if mole.contains_node(box, image):
			trim_box = mole.trimmed_box(box, image)
			node_boxes.append(trim_box)
	filtered_node_boxes = mole.filter_by_area(node_boxes)
	return set(filtered_node_boxes)

def find_edges(centroids: List[Point], ablated_image: ndarray)-> Set[Box]:
	ablated_image = advanced.threshold(ablated_image)
	edges = advanced.get_edges(centroids, ablated_image)
	graphics.draw_my_centroids(centroids, False, src_image = ablated_image)

def find_nodes(args: Dict[str, str], show: bool=False)-> List[Point]:
	imgPath = args['image']
	filter_list,  f_args_list = filters(args['filter'])
	enhance_list, e_args_list = enhancers(args['enhancer'])
	segment_list, s_args_list = segmenters(args['segmenter'])
	nodes = set()
	clone = None
	for f, f_args in zip(filter_list, f_args_list): # clone is awesome!
		for e, e_args in zip(enhance_list, e_args_list):
			for s, s_args in zip(segment_list, s_args_list):
				image, clone = mutate(imgPath, f, f_args, e, e_args, s, s_args)
				nodes = nodes.union(find_node_boxes(image, clone))
	centroids, centroid_to_box = boxer.get_centroids(list(nodes))
	condensed_centroids, box_groups = advanced.cluster_reduction(centroid_to_box, centroids)
	condensed_boxes = boxer.condense(box_groups)
	ablated_image = boxer.fill_boxes(imgPath, condensed_boxes, 0)
	expanded_boxes = boxer.expand_from_centroid(ablated_image, condensed_centroids)
	ablated_image = boxer.fill_boxes(imgPath, expanded_boxes, 255)
	if show:
		graphics.draw_my_boxes(imgPath, condensed_boxes, None)
		graphics.draw_my_centroids(condensed_centroids, True, imgPath=imgPath)
	return condensed_centroids, ablated_image


def core(args: Dict[str, str])-> None:
	centroids, ablated_image = find_nodes(args, False)
	edges = find_edges(centroids, ablated_image)
	#graphics.draw_my_boxes(args['image'], edges, None)
	exit()

if __name__ == '__main__':
	#https://pillow.readthedocs.io/en/stable/reference/Image.html
	parser = argparse.ArgumentParser()
	parser.add_argument("-i", "--image",  help="path to image", type=str)
	parser.add_argument("-f", "--filter", help="index of filter to apply to image", type=str, nargs='*', default=['nan'])
	parser.add_argument("-e", "--enhancer", help="index of enhancer to apply to image", type=str, nargs='*', default=['nan'])
	parser.add_argument("-s", "--segmenter", help="index of segmenter to apply to image", type=str, nargs='*', default=['nan'])
	args = vars(parser.parse_args())
	if not args['image']: print('\tNo image provided.'); exit()
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
