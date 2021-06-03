import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import alphashape
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.ops import cascaded_union
from shapely.geometry import Polygon


def find_intersections(polygons):
    'CONSIDER CLUSTERS SHOULD BE UNITED BASED ON PERCENTAGE OF INTERSECTION'
    'Add in dictionary whether the intersection should be united or subtracted'
    # Percentage threshold for uniting polygons
    THRESHOLD = 30
    # create empty dictionary
    intersections = dict()

    # create keys in dictionary
    for i in range(0, len(polygons)):
        key = i
        intersections[key] = []

    # Add intersections in the dictionary based on percentage criterion
    for i in range(0, len(polygons)):
        for j in range(i + 1, len(polygons)):
            intersection_percentage = []
            intersection_percentage.append((polygons[i].intersection(polygons[j]).area) / polygons[i].area * 100)
            intersection_percentage.append((polygons[i].intersection(polygons[j]).area) / polygons[j].area * 100)

            if polygons[i].intersects(polygons[j]) == True:
                if intersection_percentage[0] >= THRESHOLD or intersection_percentage[0] >= THRESHOLD:
                    key = i
                    value = [j, 'union']
                    intersections[key].append(value)
                else:
                    key = i
                    value = [j, 'subtraction']
                    intersections[key].append(value)

    return intersections


def return_unique_polygons(intersections):
  'updated with union and subtraction criteria'
  remove = [] #used to store index of keys to remove

  #check which keys in the dictionary will need to be removed
  for key in intersections:
    for value in intersections[key]:
      if value[0] in intersections:
          remove.append(value[0])

  #remove key from dictionary
  for i in range(0,len(remove)):
    #Add exception if code was trying to remove key that was already removed
    try:
      intersections.pop(remove[i])
    except KeyError:
      continue

  return intersections


def plot_new_polygons(unique_dictionary, polygons):
    'Subtracts polygons with intersection % below threshold, and combine polygons with intersection % above threshold'

    # Variable to decide whether to perform subtraction in case we have 3 or more intersecting polygons
    need_subtract = False

    for key in unique_dictionary:
        # check if the key is empty (has not values)
        if not unique_dictionary[key]:
            # plot the polygon with no intersections
            x, y = polygons[key].exterior.xy
            plt.plot(x, y)

        else:
            # create an array to add the polygons to be merged
            combination_merge = []
            # added the polygon in the key itself
            combination_merge.append(polygons[key])
            # create an array to add the polygons to be subtracted, in case there is any
            combination_substract = []

            for value in unique_dictionary[key]:
                if value[1] == 'union':
                    combination_merge.append(polygons[value[0]])

                elif value[1] == 'subtraction':
                    combination_substract.append(polygons[value[0]])
                    need_subtract = True

            # merge the polygons to be merged
            merged = cascaded_union(combination_merge)

            # If no need to subtract, then just plot the merged polygons
            if need_subtract == False:
                x, y = merged.exterior.xy
                plt.plot(x, y)

            elif need_subtract == True:
                # subtract the one to be subtracted from the merged ones
                subtracted = []
                for i in range(0, len(combination_substract)):
                    subtracted.append(merged.symmetric_difference(combination_substract[i]))
                    for j in range(0, len(subtracted[i])):
                        x, y = subtracted[i][j].exterior.xy
                        plt.plot(x, y)


def create_polygons(type, num_samples, num_clusters, random_state, *cluster_std, keep_points=False):
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
        cluster_std = 1.5 * np.random.random(num_clusters)
        data = datasets.make_blobs(n_samples=num_samples,
                                   centers=num_clusters,
                                   cluster_std=cluster_std,
                                   random_state=random_state,
                                   center_box=(-30, 30))

    plt.figure()
    plt.scatter(data[0][:, 0], data[0][:, 1], s=1)

    # Create a list of empty arrays for each cluster
    clusters = [[] for _ in range(num_clusters)]

    # Check each point to which cluster it belongs and append to the list accordingly
    for i in range(0, len(data[0])):
        cluster_index = data[1][i]
        clusters[cluster_index].append(data[0][i])

    # Create emtpy arrays for convex hulls and data points
    hulls = [[] for _ in range(num_clusters)]
    points = [[] for _ in range(num_clusters)]
    hulls_vertices = [[] for _ in range(num_clusters)]

    # Use the Concave Hull for the noisy moons shape
    if type == "noisy_moons":
        ALPHA = 5
        for i in range(0, len(clusters)):
            hull = alphashape.alphashape(np.array(clusters[i]), ALPHA)
            hull_pts = hull.exterior.coords.xy
            hulls[i] = hull_pts

        # Append vertices
        for i in range(0, len(hulls)):
            for j in range(0, len(hulls[0][i])):
                vertex = [hulls[i][0][j], hulls[i][1][j]]
                hulls_vertices[i].append(vertex)


    # Use the ConvexHull for all other shapes
    else:
        # Append the hulls
        for i in range(0, len(clusters)):
            hulls[i] = ConvexHull(clusters[i])

        # Append vertices of the hulls
        for i in range(0, len(hulls)):
            for j in range(0, len(hulls[i].vertices)):
                hulls_vertices[i].append(clusters[i][hulls[i].vertices[j]])

    # Create empty array to append the polygons
    polygons = []

    # Create polygons from hull vertices
    for i in range(0, len(hulls_vertices)):
        polygon = Polygon(np.array(hulls_vertices[i]))
        polygons.append(polygon)

    return polygons



#Test function
NUM_SAMPLES= 1500
NUM_CLUSTERS= 30

test = create_polygons(type='blobs',
                       num_samples=NUM_SAMPLES,
                       num_clusters=NUM_CLUSTERS,
                       random_state=13,
                       keep_points=False)

intersections = find_intersections(test)
dictionary = return_unique_polygons(intersections)
plt.figure()
plot_new_polygons(dictionary, test)