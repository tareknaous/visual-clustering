import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph




NUM_CLUSTERS = 10
CLUSTER_STD = 4 * np.ones(NUM_CLUSTERS)

# samples = [1000, 5000, 10000, 50000, 100000, 500000, 1000000, 1500000, 2000000]
samples = [1000, 5000, 10000, 50000]


time_visual = []
time_kmeans = []
time_dbscan = []
time_affinity = []
time_spectral = []
time_optics = []
time_gmm = []
time_ward = []
time_ms = []
time_birch = []
time_agglo = []



for i in samples:
  data = datasets.make_blobs(n_samples=i, centers=NUM_CLUSTERS, random_state=151,center_box=(0, 256), cluster_std=CLUSTER_STD)

  #Compute Visual
  start = time.time()
  input = create_input_image(data)
  result = predict_sample(input)
  y_km = get_instances(result, data)
  end = time.time()

  time_visual.append(end-start)

  #Compute Kmeans
  km = KMeans(
    n_clusters=10, init='random',
    n_init=10, max_iter=300, 
    tol=1e-04, random_state=0
  )

  start = time.time()
  y_km = km.fit_predict(data[0])
  end = time.time()
  time_kmeans.append(end - start)

  #Compute dbscan
  dbscan = cluster.DBSCAN(eps=0.15)

  start = time.time()
  y_km = dbscan.fit_predict(data[0])
  end = time.time()
  time_dbscan.append(end - start)

  #Compute Affinity Propagation
  # affinity_propagation = cluster.AffinityPropagation(damping=0.77, preference=-240)

  # start = time.time()
  # y_km = affinity_propagation.fit_predict(data[0])
  # end = time.time()
  # time_affinity.append(end - start)

  #Compute Spectral
  spectral = cluster.SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity="nearest_neighbors")
  start = time.time()
  y_km = dbscan.fit_predict(data[0])
  end = time.time()
  time_spectral.append(end - start)

  #Compute OPTICS
  # optics = cluster.OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.2)
  # start = time.time()
  # y_km = optics.fit_predict(data[0])
  # end = time.time()
  # time_optics.append(end - start)

  #Compute GMM
  gmm = mixture.GaussianMixture(n_components=10, covariance_type='full')
  start = time.time()
  y_km = gmm.fit_predict(data[0])
  end = time.time()
  time_gmm.append(end - start)  

  #Ward
  # start = time.time()
  # # connectivity matrix for structured Ward
  # connectivity = kneighbors_graph(data[0], n_neighbors=10, include_self=False)
  # # make connectivity symmetric
  # connectivity = 0.5 * (connectivity + connectivity.T)
  # ward = cluster.AgglomerativeClustering(n_clusters=10, linkage='ward', connectivity=connectivity)
  # y_km = ward.fit_predict(data[0])
  # end = time.time()
  # time_ward.append(end - start)  

  #Mean Shift
  # estimate bandwidth for mean shift
  # start = time.time()
  # bandwidth = cluster.estimate_bandwidth(data[0], quantile=0.2)  
  # ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
  # y_km = ms.fit_predict(data[0])
  # end = time.time()
  # time_ms.append(end - start)  

  #Birch
  # birch = cluster.Birch(n_clusters=10)
  # start = time.time()
  # y_km = birch.fit_predict(data[0])
  # end = time.time()
  # time_birch.append(end - start)  

  #Agglomertive 
  connectivity = kneighbors_graph(data[0], n_neighbors=10, include_self=False)
  # make connectivity symmetric
  connectivity = 0.5 * (connectivity + connectivity.T)
  average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock",n_clusters=10, connectivity=connectivity)
  start = time.time()
  y_km = average_linkage.fit_predict(data[0])
  end = time.time()
  time_agglo.append(end - start)  

plt.plot(samples, time_visual , 'r', marker='o', linewidth=2)
plt.plot(samples, time_kmeans, 'b', marker='o',linewidth=2)
plt.plot(samples, time_dbscan, 'g', marker='o',linewidth=2)
plt.plot(samples, time_spectral, 'm', marker='o',linewidth=2)
plt.plot(samples, time_gmm, 'y', marker='o',linewidth=2)
plt.plot(samples, time_agglo, 'c', marker='o',linewidth=2)




plt.legend(['Our Method', 'K-Means', 'DBSCAN', 'Spectral Clustering', 'Gaussian Mixture', 'Agglomerative Clustering'], fontsize=11)
plt.grid( linestyle='-', linewidth=0.5)
plt.ylabel('Time (sec)', fontsize=16)
plt.xlabel('Samples', fontsize=16)
plt.title("Time vs Nb of Samples for 10 blobs")




