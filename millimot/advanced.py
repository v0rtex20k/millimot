import cv2
import numpy as np
from math import sqrt
from PIL import ImageDraw
from PIL import Image as pillow
from collections import Counter
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict, Set
from matplotlib import pyplot as mtplt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

ndarray = List
Box = Tuple[int, int, int, int]
Point = Tuple[int, int]


def make_box_groups(row, centroid_to_box, mini_group):
	mini_group.append(centroid_to_box[(row[0],row[1])])

def cluster_reduction(centroid_to_box: Dict[Point, Box], centroids: List[Point])-> Tuple[List[Point], List[ndarray[Box]]]:
	X = np.asarray(centroids)
	start, stop = 5, 55 # arbitrary, but seems to work well
	silhouettes, labels = np.zeros(stop+1-start), []
	for i in range(start, stop+1, 1):
		kmeans = KMeans(n_clusters=i).fit(X)
		y_pred = kmeans.predict(X)
		labels.append(kmeans.labels_)
		silhouettes[i-start] = silhouette_score(X, y_pred)

	best_labels = labels[np.argmax(silhouettes)] # highest silhouette score
	real_clusters = np.asarray([k for k,v in Counter(best_labels).items() if v >= 3]) # only keep clusters with at least 3 nearby centroids
	confirmed_labels = np.isin(best_labels, real_clusters) # only keep labels with at least 3 nearby centroids

	confirmed_centroids = np.c_[X[confirmed_labels], best_labels[confirmed_labels]]
	condensed_centroids = np.zeros((len(real_clusters), 2))

	current_row, box_groups = 0, []
	for label in real_clusters:
		mini_group = []
		cluster = confirmed_centroids[confirmed_centroids[:, 2] == label]
		condensed_centroids[current_row, :] = np.mean(cluster[:,:2], axis=0)
		np.apply_along_axis(make_box_groups, 1, cluster, centroid_to_box, mini_group)
		current_row += 1
		assert(len(mini_group) == cluster.shape[0]) # must be as many boxes as centroids in cluster
		box_groups.append(np.asarray(mini_group))
	return [(int(x),int(y)) for x,y in condensed_centroids.tolist()], box_groups

N = lambda r, c: [(r-1, c), (r+1, c), (r, c-1), (r, c+1), (r+1, c-1), (r+1, c+1), (r-1, c-1), (r-1, c+1)]
def threshold(ablated_image: ndarray)-> ndarray:
	image_arr = np.asarray(ablated_image).copy()
	image_arr[image_arr <= 100]  = 0
	image_arr[image_arr > 100] = 255
	return pillow.fromarray(image_arr)

def build_edgelist(endpoints_to_centroids: Dict[Point, Point], endpoint_pairs: Dict[Point, Point])-> Set[Tuple[Point, Point]]:
	edgelist = set()
	for end1, end2 in endpoint_pairs.items():
		if end1 == end2: continue
		c1, c2 = endpoints_to_centroids[end1], endpoints_to_centroids[end2]
		if c1 == c2: continue
		edgelist.add((c1, c2))
	return edgelist

def map_endpoints(row: ndarray, centroids: ndarray, endpoints: ndarray, endpoints_to_centroids: Dict[Point, Point])-> None:
	xe, ye = endpoints[row[0], :]
	xc, yc = centroids[row[1], :]
	endpoints_to_centroids[(xe, ye)] = (xc, yc)

def get_endpoints(x: int, y: int, image_arr: ndarray)-> Tuple[Point, Point]:
	visited, queue  = {(x,y)}, {*[(r,c) for r,c in N(y,x) if image_arr[r,c] == 0]}
	if not queue:
		image_arr[y,x] = 255
		return None
	while queue:
		y2,x2 = queue.pop()
		if image_arr[y2,x2] == 0:
			visited.add((x2,y2))
			for r,c in N(y2,x2):
				if image_arr[r,c] == 0:
					queue.add((r,c))
			image_arr[y2,x2] = 255 # set to white so I don't explore the same place again
	visited = np.array(list(visited))
	distances = cdist(visited, visited, 'euclidean')
	i, j = np.unravel_index(np.argmax(distances), distances.shape)
	return tuple(visited[i]), tuple(visited[j]) # stored as (x,y)


def get_edges(centroids: List[Point], ablated_image: ndarray)-> List[Tuple[Point, Point]]:
	image_arr = np.asarray(ablated_image).copy()
	endpoint_pairs = dict()
	loose_endpoints = []
	while image_arr[image_arr == 0].size > 0: # not exactly zero - might be some odd fragments
		black_pixels = np.argwhere(image_arr == 0)
		source = np.random.randint(black_pixels.shape[0])
		y, x = black_pixels[source, :]
		if image_arr[y,x] == 0:
			endpoints = get_endpoints(x, y, image_arr)
			if endpoints:
				(x1, y1), (x2, y2) = endpoints
				if sqrt((x2 - x1)**2 + (y2 - y1)**2) <= 10:
					continue
				endpoint_pairs[(x1,y1)] = (x2, y2)
				endpoint_pairs[(x2,y2)] = (x1, y1)
				loose_endpoints.extend([(x1, y1), ((x1+x2)//2, (y1+y2)//2) ,(x2, y2)])

	#endpoints_to_centroids = dict()
	#loose_endpoints = np.array(loose_endpoints).reshape(-1,2)
	#centroids = np.array(centroids)
	#distances = cdist(loose_endpoints, centroids, 'euclidean')
	#nearby_centroids = np.c_[np.arange(loose_endpoints.shape[0]), np.argmin(distances, axis=1)]
	#np.apply_along_axis(map_endpoints, 1, nearby_centroids, centroids, loose_endpoints, endpoints_to_centroids)
	#edgelist = build_edgelist(endpoints_to_centroids, endpoint_pairs)
	#artist = ImageDraw.Draw(ablated_image)
	blank = pillow.fromarray(np.full_like(ablated_image, 0))
	artist = ImageDraw.Draw(blank)
	for edge in endpoint_pairs.items():
		artist.line(edge, fill=255)
	for centroid in centroids:
		x, y = centroid
		r = 3
		artist.ellipse([(x-r, y-r), (x+r, y+r)], fill=200)
	blank.show()
	ablated_image.show()
	exit()
	return loose_endpoints


