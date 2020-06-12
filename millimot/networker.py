import numpy as np
from math import hypot
from PIL import ImageDraw
from PIL import Image as pillow
from collections import OrderedDict
from typing import List, Tuple, Mapping

Box = Tuple[int, int, int, int]
Node_Frame_Dict = Mapping[Box, Tuple[int, Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]

# Each rectangle becomes 4 line segments:
#	l1 = [(x,y), (x+w,y)]  		- (up)
#	l2 = [(x,y), (x,y+h)]  		| (left)
#	l3 = [(x+w,y), (x+w,y+h)]   | (right)    
#	l4 = [(x,y+h), (x+w,y+h)]   - (down)

# rank rectangles by euclidean distance of midpoint of line to 
# center of rectangle [x+(w//2), y+(h//2)]

def distance(p1: Tuple[int, int], p2: Tuple[int, int])-> float:
	x1, y1 = p1
	x2, y2 = p2
	return hypot(x2 - x1, y2 - y1)

def get_rectangle_center(box: Box)-> Tuple[int, int]:
	x, y, w, h = box
	return [x+(w//2), y+(h//2)]

def midpoint(p1: Tuple[int, int], p2: Tuple[int, int])-> Tuple[int, int]:
	x1, y1 = p1
	x2, y2 = p2
	return ((x1+x2)//2, (y1+y2)//2)

def encode_line(p1: Tuple[int, int], p2: Tuple[int, int])-> Tuple[int, int, int]:
	A = (p1[1] - p2[1])
	B = (p2[0] - p1[0])
	C = (p1[0]*p2[1] - p2[0]*p1[1])
	return A, B, -C

def intersection(L1: Tuple[int, int, int], L2: Tuple[int, int, int])-> bool:
	D  = L1[0] * L2[1] - L1[1] * L2[0]
	Dx = L1[2] * L2[1] - L1[1] * L2[2]
	Dy = L1[0] * L2[2] - L1[2] * L2[0]
	if D != 0: # AND IN BOUNDS
		return Dx / D, Dy / D
	return None, None

def frames(box: Box)-> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
	x, y, w, h = box
	up    = ((x,y), (x+w,y))  		# - (up)
	down  = ((x,y+h), (x+w,y+h))    # - (down)
	left  = ((x,y), (x,y+h))  		# | (left)
	right = ((x+w,y), (x+w,y+h))    # | (right)
	borders = [up, down, left, right]
	return [encode_line(*border) for border in borders]

def rank_nodes(midpoint: Tuple[int, int], nodes: List[Box]):
	sorted_nodes = {rect: (distance(midpoint, get_rectangle_center(rect)), frames(rect)) for rect in nodes}
	sorted_nodes = OrderedDict({k: v for k, v in sorted(sorted_nodes.items(), key=lambda item: item[1][0])})
	return sorted_nodes

def enclosed(edge: Tuple[Tuple[int, int], Tuple[int, int]], box: Box)-> bool:
	x, y, w, h = box
	(x1, y1), (x2, y2) = edge
	if x <= x1 <= w and y <= y1 <= h:
		return True
	elif x <= x2 <= w and y <= y2 <= h:
		return True
	return False

def find_valid_intersections(edge: Tuple[int, int], sorted_node_frames: Node_Frame_Dict)-> [List[Box], List[Box]]:
	# W AND H INCLUDE X AND Y
	pad = 5
	line = encode_line(*edge)
	incident_nodes, intersections = [], []
	for rect in sorted_node_frames.keys():
		if enclosed(edge, rect): return None, None
	for rect, (dist, frame) in sorted_node_frames.items():
		for i, encoded_border in enumerate(frame):
			x, y = intersection(line, encoded_border)
			if x is None and y is None: continue # find first intersection
			# up0 down1 left2 right3
			rx, ry, rw, rh = rect
			if i in [0,1] and rx-pad <= x <= rw+pad: # pad can't be out of bounds b/c we're not indexing!!!!
				incident_nodes.append(rect)
				intersections.append((x, y, i))
				if len(incident_nodes) == 2:
					return tuple(incident_nodes), intersections
				break
			elif i in [2,3] and ry-pad <= y <= rh+pad:
				incident_nodes.append(rect)
				intersections.append((x, y, i))
				if len(incident_nodes) == 2:
					return tuple(incident_nodes), intersections
				break
	return None, None # could make something that just guesses second nearest ... worth considering. For now only keep confirmed ones.
	
	