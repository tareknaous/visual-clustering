from utils.functions import create_polygons
from utils.functions import find_intersections
from utils.functions import return_unique_polygons
from utils.functions import plot_new_polygons
from utils.functions import create_mask


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import alphashape
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.ops import cascaded_union
from shapely.geometry import Polygon


#Test function
NUM_SAMPLES= 1500
NUM_CLUSTERS= 11
test = create_polygons(type='blobs',
                       num_samples=NUM_SAMPLES,
                       num_clusters=NUM_CLUSTERS,
                       random_state=19,
                       keep_points=True)

intersections = find_intersections(test)
dictionary = return_unique_polygons(intersections)
plt.savefig('cluster.png')
polygons = plot_new_polygons(dictionary, test)
plt.figure()
create_mask(polygons)
plt.savefig('annotation.png')