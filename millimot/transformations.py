import cv2
import numpy as np
import scipy.ndimage as sn
from typing import List, Callable, Tuple

ndarray = List
Transformer = (Callable[[ndarray, int], ndarray], Tuple)

def myCanny(img: ndarray, *args: Tuple[int])-> ndarray:
	return cv2.Canny(np.asarray(img), *args)

def mean_filter(img: ndarray, size: int) -> ndarray:
	return sn.convolve(img, np.ones((size,size))/(size**2))

def filters(keys: str)-> Transformer:
	F = {'nan': (None, None),
		 'sob': (sn.sobel, ()),
		 'prw': (sn.prewitt, ()),
	     'lap': (sn.laplace, ()),
		 'mew': (mean_filter, (5,)),
		 'cny': (myCanny, (100, 200)), 
		 'med': (sn.median_filter, (5,)), 
	     'max': (sn.maximum_filter, (5,)), 
	     'min': (sn.minimum_filter, (5,)), 
	     'LoG': (sn.gaussian_laplace, (1,))}
	filter_funcs = []
	try:
		for key in keys:
			filter_funcs.append(F[key])
	except KeyError:
		print("[FILTER]\t ---> \"{}\" is not a recognized filter.".format(key)); exit(1)
	return ([f[0] for f in filter_funcs], [f[1] for f in filter_funcs])

def inverse(img: ndarray)-> ndarray:
	img_array = np.asarray(img)
	return 255-img_array

def logarithmic(img: ndarray)-> ndarray:
	img_array = np.asarray(img).astype(float)
	max_pixel = np.amax(img_array)
	imgT = (255.0 * np.log(1.0+img_array))/(np.log(1.0+max_pixel))
	return imgT.astype(int)

def power_law(img: ndarray, gamma: float)-> ndarray:
	img_array = np.asarray(img).astype(float)
	norm_img_array = np.where(img_array/np.amax(img_array) > 0, img_array/np.amax(img_array), 1) # avoid /0 error
	gamma_exp = np.log(norm_img_array)*gamma
	imgT = np.exp(gamma_exp)*255.0
	return imgT.astype(int)

def histogram_equalization(img: ndarray)-> ndarray:
	img_array = np.asarray(img)
	max_pixel = np.amax(img_array)
	hist, bins = np.histogram(img_array, 256, [0,255])
	cdf = hist.cumsum()
	cdf_masked = np.ma.masked_equal(cdf, 0)
	cdf_masked = ((cdf_masked - np.amin(cdf_masked))*255)/(np.amax(cdf_masked) - np.amin(cdf_masked))
	cdf = np.ma.filled(cdf_masked, 0)
	return np.reshape(cdf[img_array.flatten()], img_array.shape)

def enhancers(keys: str)-> Transformer:
	E = {'nan': (None, None),
		 'inv': (inverse, ()),
		 'log': (logarithmic, ()), # compatible w/ wtr
		 'pwr': (power_law, (0.5,)),
		 'heq': (histogram_equalization, ())} # compatible w/ wtr
	enhance_funcs = []
	try:
		for key in keys:
			enhance_funcs.append(E[key])
	except KeyError:
		print("[ENHANCER]\t ---> \"{}\" is not a recognized enhancer.".format(key)); exit(1)
	return ([e[0] for e in enhance_funcs], [e[1] for e in enhance_funcs])
