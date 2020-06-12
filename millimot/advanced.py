import cv2
import numpy as np
from PIL import ImageDraw
from PIL import Image as pillow
from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import pyplot as mtplt
from typing import List, Callable, Tuple, NewType
from sklearn.metrics import silhouette_score # global mean

ndarray, Point, Box = List, Tuple[int, int], Tuple[int, int, int, int]

def cluster_reduction(nodes: List[Box], centroids: List[Point])-> List[Box]:
	X = np.asarray(centroids)
	start, stop = 5, 55
	silhouettes, labellings = np.zeros(stop+1-start), []
	for i in range(start, stop+1, 1):
		kmeans = KMeans(n_clusters=i).fit(X)
		y_pred = kmeans.predict(X)
		labellings.append(kmeans.labels_)
		silhouettes[i-start] = silhouette_score(X, y_pred)

	print('Best number of clusters: ', np.argmax(silhouettes)+start)
	labels = labellings[np.argmax(silhouettes)]

	real_clusters = np.asarray([k for k,v in Counter(labels).items() if v >= 3])
	valid_labels = np.isin(labels, real_clusters)

	labeled_centroids = np.c_[X[valid_labels], labels[valid_labels]]
	condensed_centroids = np.zeros((len(real_clusters), 2))

	current_row = 0
	for label in real_clusters:
		cluster = labeled_centroids[labeled_centroids[:, 2] == label]
		condensed_centroids[current_row, :] = np.mean(cluster[:,:2], axis=0)
		current_row += 1
	return [(x,y) for x,y in condensed_centroids.tolist()]



