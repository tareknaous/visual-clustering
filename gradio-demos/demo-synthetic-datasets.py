import gradio as gr
from itertools import cycle, islice


def visual_clustering(cluster_type, num_clusters, num_samples, random_state, median_kernel_size, max_kernel_size):

  NUM_CLUSTERS = num_clusters
  CLUSTER_STD = 4 * np.ones(NUM_CLUSTERS)

  if cluster_type == "blobs":
    data = datasets.make_blobs(n_samples=num_samples, centers=NUM_CLUSTERS, random_state=random_state,center_box=(0, 256), cluster_std=CLUSTER_STD)
  
  elif cluster_type == "varied blobs":
    cluster_std = 1.5 * np.ones(NUM_CLUSTERS)
    data = datasets.make_blobs(n_samples=num_samples, centers=NUM_CLUSTERS, cluster_std=cluster_std, random_state=random_state)

  elif cluster_type == "aniso":
    X, y = datasets.make_blobs(n_samples=num_samples, centers=NUM_CLUSTERS, random_state=random_state, center_box=(-30, 30))
    transformation = [[0.8, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    data = (X_aniso, y)

  elif cluster_type == "noisy moons":
    data = datasets.make_moons(n_samples=num_samples, noise=.05)

  elif cluster_type == "noisy circles":
    data = datasets.make_circles(n_samples=num_samples, factor=.01, noise=.05)

  max_x = max(data[0][:, 0])
  min_x = min(data[0][:, 0])
  new_max = 256
  new_min = 0

  data[0][:, 0] = (((data[0][:, 0] - min_x)*(new_max-new_min))/(max_x-min_x))+ new_min

  max_y = max(data[0][:, 1])
  min_y = min(data[0][:, 1])
  new_max_y = 256
  new_min_y = 0

  data[0][:, 1] = (((data[0][:, 1] - min_y)*(new_max_y-new_min_y))/(max_y-min_y))+ new_min_y

  fig1 = plt.figure()
  plt.scatter(data[0][:, 0], data[0][:, 1], s=1, c='black')
  plt.close()

  input = create_input_image(data[0])
  filtered = ndimage.median_filter(input, size=median_kernel_size)
  result = predict_sample(filtered)
  y_km = get_instances(result, data[0], max_filter_size=max_kernel_size)

  colors = np.array(list(islice(cycle(["#000000", '#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00' ,'#491010']),
                                      int(max(y_km) + 1))))
  #add black color for outliers (if any)
  colors = np.append(colors, ["#000000"])
  
  fig2 = plt.figure()
  plt.scatter(data[0][:, 0], data[0][:, 1], s=10, color=colors[y_km.astype('int8')])
  plt.close()

  return fig1, fig2

iface = gr.Interface(
    
  fn=visual_clustering, 

  inputs=[
          gr.inputs.Dropdown(["blobs", "varied blobs",  "aniso", "noisy moons", "noisy circles" ]),
          gr.inputs.Slider(1, 10, step=1, label='Number of Clusters'),
          gr.inputs.Slider(10000, 1000000, step=10000, label='Number of Samples'),
          gr.inputs.Slider(1, 100, step=1, label='Random State'),
          gr.inputs.Slider(1, 100, step=1, label='Denoising Filter Kernel Size'),
          gr.inputs.Slider(1,100, step=1, label='Max Filter Kernel Size')
          ],

  outputs=[
           gr.outputs.Image(type='plot', label='Dataset'),
           gr.outputs.Image(type='plot', label='Clustering Result')
           ]
           )
iface.launch(share=True)