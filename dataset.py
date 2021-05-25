import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

#blobs
n_samples = 1500
blobs = datasets.make_blobs(n_samples=n_samples, centers=4, random_state=3)
# plt.scatter(blobs[0][:,0],blobs[0][:,1])
# plt.show()

cluster_0_points = []
cluster_1_points = []
cluster_2_points = []
cluster_3_points = []

for i in range(0,len(blobs[0])):
  if blobs[1][i] == 0:
    cluster_0_points.append(blobs[0][i])
  if blobs[1][i] == 1:
    cluster_1_points.append(blobs[0][i])
  if blobs[1][i] == 2:
    cluster_2_points.append(blobs[0][i])
  if blobs[1][i] == 3:
    cluster_3_points.append(blobs[0][i])


clusters = []

clusters.append(cluster_0_points)
clusters.append(cluster_1_points)
clusters.append(cluster_2_points)
clusters.append(cluster_3_points)



from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

#Cluster 0
hull_0 = ConvexHull(cluster_0_points)
points_0 = np.array(cluster_0_points)

for simplex in hull_0.simplices:
    plt.plot(points_0[simplex, 0], points_0[simplex, 1], 'k-')



#Cluster 1
hull_1 = ConvexHull(cluster_1_points)
points_1 = np.array(cluster_1_points)

for simplex in hull_1.simplices:
    plt.plot(points_1[simplex, 0], points_1[simplex, 1], 'k-')


#Cluster 2
hull_2 = ConvexHull(cluster_2_points)
points_2 = np.array(cluster_2_points)

for simplex in hull_2.simplices:
    plt.plot(points_2[simplex, 0], points_2[simplex, 1], 'k-')



#Cluster 3
hull_3 = ConvexHull(cluster_3_points)
points_3 = np.array(cluster_3_points)

for simplex in hull_3.simplices:
    plt.plot(points_3[simplex, 0], points_3[simplex, 1], 'k-')


plt.show()