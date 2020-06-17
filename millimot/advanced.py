import cv2
import numpy as np
from PIL import ImageDraw
from PIL import Image as pillow
from collections import Counter
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
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
	image_arr[image_arr < 100]  = 0
	image_arr[image_arr >= 100] = 255
	return pillow.fromarray(image_arr)

def get_endpoints(x: int, y: int, image_arr: ndarray)-> Tuple[Point, Point]:
	visited, queue  = {(x,y)}, {*[(r,c) for r,c in N(y,x) if image_arr[r,c] == 0]}
	while queue:
		y2,x2 = queue.pop()
		if image_arr[y2,x2] == 0 and image_arr[y2,x2] not in visited:
			image_arr[y2,x2] = 255 # set to white so we don't explore the same place again
			visited.add((x2,y2))
			queue.union(set([(r,c) for r,c in N(y2,x2) if image_arr[r,c] == 0 and (c,r) not in visited]))
	visited = np.array(list(visited))
	distances = cdist(visited, visited, 'euclidean')
	i, j = np.unravel_index(np.argmax(distances), distances.shape)
	return (tuple(visited[i]), tuple(visited[j]))


def get_edges(centroids: List[Point], ablated_image: ndarray)-> List[Tuple[Point, Point]]:
	# randomly choose black point
	# get all connected black points
	# find farthest points in minigraph
	# find closest centroid to each endpoint and return edge
	edgelist = set()
	image_arr = np.asarray(ablated_image).copy()
	endpoint_pairs, centroids = set(), np.array(centroids)
	while image_arr[image_arr == 0].size > 0:
		m,n = image_arr.shape
		x = np.random.randint(n)
		y = np.random.randint(m)
		if image_arr[y,x] == 0:
			endpoints = get_endpoints(x, y, image_arr)
			endpoint_pairs.add(endpoints)
	for pair in endpoint_pairs:
		pair = np.array(pair)
		distances = cdist(pair, centroids, 'euclidean')
		# find minimum for each endpoint, then make that an edge and add to edgelist.
		exit()
		#i, j = np.unravel_index(np.argmin(distances), distances.shape)
		#edgelist.add((cent))







