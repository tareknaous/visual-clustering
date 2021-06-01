import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import alphashape
from shapely.geometry import Polygon
from shapely.ops import cascaded_union
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def plot_polygons(type, num_samples, num_clusters, random_state, *cluster_std, keep_points=False):
    if type == 'blobs':  # works fine
        data = datasets.make_blobs(n_samples=num_samples, centers=num_clusters, random_state=random_state,
                                   center_box=(-30, 30))

    if type == 'aniso':  # works fine
        X, y = datasets.make_blobs(n_samples=num_samples, centers=num_clusters, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X_aniso = np.dot(X, transformation)
        data = (X_aniso, y)

    if type == 'noisy_moons':  # works fine
        data = datasets.make_moons(n_samples=num_samples, noise=.05)
        if num_clusters != 2:
            raise Exception("Can only take 2 clusters for noisy_moons")

    if type == 'noisy_circles':  # works fine
        data = datasets.make_circles(n_samples=num_samples, factor=.01, noise=.2)
        if num_clusters != 2:
            raise Exception("Can only take 2 clusters for noisy_circles")

    if type == 'varied_blobs':  # works fine
        cluster_std = np.random.random(num_clusters)
        data = datasets.make_blobs(n_samples=num_samples,
                                   centers=num_clusters,
                                   cluster_std=cluster_std,
                                   random_state=random_state,
                                   center_box=(-30, 30))

    plt.figure()
    plt.scatter(data[0][:, 0], data[0][:, 1])

    # Create a list of empty arrays for each cluster
    clusters = [[] for _ in range(num_clusters)]

    # Check each point to which cluster it belongs and append to the list accordingly
    for i in range(0, len(data[0])):
        cluster_index = data[1][i]
        clusters[cluster_index].append(data[0][i])

    # Create emtpy arrays for convex hulls and data points
    hulls = [[] for _ in range(num_clusters)]
    points = [[] for _ in range(num_clusters)]

    # Use the Concave Hull
    if type == "noisy_moons":
        ALPHA = 5
        for i in range(0, len(clusters)):
            hull = alphashape.alphashape(np.array(clusters[i]), ALPHA)
            hull_pts = hull.exterior.coords.xy
            hulls[i] = hull_pts
            points[i].append(np.array(clusters[i]))

        plt.figure()
        for i in range(0, len(clusters)):
            plt.plot(hulls[i][0], hulls[i][1], color='black')
            if keep_points == True:
                plt.plot(points[i][0][:, 0], points[i][0][:, 1], 'o')

                # Use the ConvexHull
    else:
        for i in range(0, len(clusters)):
            hulls[i] = ConvexHull(clusters[i])
            points[i].append(np.array(clusters[i]))

        # Plot the polygon
        plt.figure()
        for i in range(0, len(clusters)):
            for simplex in hulls[i].simplices:
                plt.plot(points[i][0][simplex, 0], points[i][0][simplex, 1], 'k-')
                if keep_points == True:
                    plt.plot(points[i][0][:, 0], points[i][0][:, 1], 'o')  # Plot points


def find_intersections(polygons):
'CONSIDER CLUSTERS SHOULD BE UNITED BASED ONLY IF THEY INTERSECT'
    # create empty dictionary
    intersections = dict()
    # create keys in dictionary
    for i in range(0, len(polygons)):
        key = i
        intersections[key] = []

    for i in range(0, len(polygons)):
        for j in range(i + 1, len(polygons)):
            if polygons[i].intersects(polygons[j]):
                key = i
                value = j
                intersections[key].append(value)

    return intersections

def return_unique_polygons(intersections):
  remove = []
  for key in intersections:
    for value in intersections[key]:
      if value in intersections:
        remove.append(value)

  for i in range(0,len(remove)):
    intersections.pop(remove[i])

  return intersections


def plot_new_polygons(unique_dictionary, polygons):
  for key in unique_dictionary:
    if not unique_dictionary[key]:
      x,y = polygons[key].exterior.xy
      plt.plot(x,y)
    else:
      combination = []
      combination.append(polygons[key])
      for value in unique_dictionary[key]:
        combination.append(polygons[value])
      merged = cascaded_union(combination)
      x,y = merged.exterior.xy
      plt.plot(x,y)

