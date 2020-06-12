import cv2
import numpy as np
import scipy.ndimage as sn
from typing import List, Callable, Tuple

ndarray = List
Segmenter = (Callable[[ndarray, int], ndarray], Tuple)

def segmenters(keys: str)-> Segmenter:
	S = {'nan': (None, None),
		 'toz': (cv2.threshold, (127,255,cv2.THRESH_TOZERO)),
		 'trc': (cv2.threshold, (127, 255, cv2.THRESH_TRUNC)),
		 'bin': (cv2.threshold, (127, 255, cv2.THRESH_BINARY)), # fucked up
		 'tzi': (cv2.threshold, (127,255,cv2.THRESH_TOZERO_INV)), # fucked up
		 'biv': (cv2.threshold, (127, 255, cv2.THRESH_BINARY_INV)), # fucked up
		 'ots': (cv2.threshold, (127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)),
		 'amu': (cv2.adaptiveThreshold, (255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)), # must be bin or biv
		 'ags': (cv2.adaptiveThreshold, (255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))}
	seg_funcs = []
	for key in keys:
		try:
			seg_funcs.append(S[key])
		except KeyError:
			print("[SEGMENTER]\t ---> \"{}\" is not a recognized segmenter.".format(key)); exit(1)
	return ([s[0] for s in seg_funcs], [s[1] for s in seg_funcs])