import cv2
import numpy as np
from PIL import ImageDraw
from PIL import Image as pillow
from collections import Counter
from sklearn.cluster import KMeans
from typing import List, Tuple, Dict
from matplotlib import pyplot as mtplt
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


'''def erode(ablated_image: ndarray)-> ndarray:
    bgr_cv2_image  = cv2.cvtColor(np.asarray(ablated_image), cv2.COLOR_GRAY2BGR)
    gray_cv2_image = cv2.cvtColor(bgr_cv2_image, cv2.COLOR_BGR2GRAY)
    vd_kernel = np.uint8([[1,0,0,0,1,0,0,0,1], # vertical & diagonal
                          [0,1,0,0,1,0,0,1,0],
                          [0,0,0,1,1,1,0,0,0],
                          [0,0,0,0,1,0,0,0,0],
                          [0,0,0,1,1,1,0,0,0],
                          [0,0,1,0,1,0,1,0,0],
                          [1,0,0,0,1,0,0,0,1]])

    eroded_gray_cv2_image = cv2.erode(gray_cv2_image, vd_kernel, iterations=1)
    #eroded_gray_cv2_image = cv2.erode(eroded_gray_cv2_image, v_kernel, iterations=1)
    eroded_arr = np.uint8(eroded_gray_cv2_image)
    eroded_arr[eroded_arr < 100] = 0
    eroded_arr[eroded_arr > 100] = 255
    return pillow.fromarray(eroded_arr)

def follow_paths(source: Point, eroded_image: ndarray)-> List[Point]:
	arr = np.asarray(eroded_image)
	explored, to_be_explored = [], []
	explored.append(source)
	# go until you hit a gray pixel, then quit - you can find closest centroid later and append that.

def get_edgelist(centroids: List[Point], eroded_image: ndarray)-> List[Tuple[Point, Point]]:
	centroids = sorted(centroids, key= lambda p: p[1])
	for centroid in centroids:
		follow_paths(centroid, eroded_image)'''



