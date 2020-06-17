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
	image_arr[image_arr <= 200]  = 0
	image_arr[image_arr > 200] = 255
	return pillow.fromarray(image_arr)

def build_edgelist(centroids_to_endpoints: Dict[Point, Set[Point]], endpoint_pairs: Dict[Point, Point])-> Set[Tuple[Point, Point]]:
	edgelist = set()
	for curr in centroids_to_endpoints:
		for other in centroids_to_endpoints:
			if curr == other: continue
			for endpoint in centroids_to_endpoints[curr]:
				if endpoint_pairs[endpoint] in centroids_to_endpoints[other] and endpoint not in centroids_to_endpoints[other]:
					edgelist.add((curr, other))
	return edgelist

def map_endpoints(row: ndarray, centroids: ndarray, endpoints: ndarray, centroids_to_endpoints: Dict[Point, Set[Point]])-> None:
	xc, yc = centroids[row[0], :]
	xe, ye = endpoints[row[1], :]
	centroids_to_endpoints[(xc, yc)].add((xe, ye))

def get_endpoints(x: int, y: int, image_arr: ndarray)-> Tuple[Point, Point]:
	visited, queue  = {(x,y)}, {*[(r,c) for r,c in N(y,x) if image_arr[r,c] == 0]}
	if not queue:
		image_arr[y,x] = 255
		return None
	while queue:
		#print(len(queue), end=' ... ')
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
	centroids_to_endpoints = {c: set() for c in centroids}
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
				loose_endpoints.extend(endpoints)
	loose_endpoints = np.array(loose_endpoints).reshape(-1,2)
	centroids = np.array(centroids)
	distances = cdist(centroids, loose_endpoints, 'euclidean')
	nearby_endpoints = np.argwhere(distances <= 150)
	np.apply_along_axis(map_endpoints, 1, nearby_endpoints, centroids, loose_endpoints, centroids_to_endpoints)
	edgelist = build_edgelist(centroids_to_endpoints, endpoint_pairs)
	artist = ImageDraw.Draw(ablated_image)
	for endpoint in loose_endpoints:
		artist.point(endpoint, fill=150)
	for edge in edgelist:
		artist.line(edge, fill=150)
	#ablated_image.show()
	#exit()
	return loose_endpoints
# https://tufts.zoom.us/j/94880736374
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4908361/pdf/btw277.pdf
	'''
	for pair in endpoint_pairs:
		p1, p2 = np.array(pair[0]), np.array(pair[1])
		if np.linalg.norm(p1-p2) < 15: # bad to have a hard limit ... look at bimodal hist? adaptive threshold
			continue
		distances = cdist(np.array(pair), centroids, 'euclidean')
		if any(np.amin(distances, axis=1) > 200): # bad to have a hard limit ... look at bimodal hist? adaptive threshold
			continue
		incident_nodes = centroids[np.argmin(distances, axis=1)]
		(xs, ys), (xd, yd) = incident_nodes[0,:], incident_nodes[1,:]
		if (xs, ys) == (xd, yd):
			continue
		edgelist.add(((xs, ys), (xd, yd)))
	'''
	exit()
	#blank = pillow.fromarray(np.full_like(ablated_image, 255))
	artist = ImageDraw.Draw(ablated_image)
	#dummy = ImageDraw.Draw(blank)
	for edge in edgelist:
		print(edge)
		artist.line(edge, fill=150)
	ablated_image.show()



