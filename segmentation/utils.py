import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from tensorflow.keras.preprocessing import image
import math
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

#Function that predicts on only 1 sample 
def predict_sample(image):
  prediction = model.predict(image[tf.newaxis, ...])
  prediction[prediction > 0.5 ] = 1
  prediction[prediction !=1] = 0
  result = prediction[0]*255
  return result





def create_input_image(data, visualize=False):
  #Initialize input matrix
  input = np.ones((256,256))

  #Fill matrix with data point values
  for i in range(0,len(data[0])):
    if math.floor(data[0][i][0]) < 256 and math.floor(data[0][i][1]) < 256:
      input[math.floor(data[0][i][0])][math.floor(data[0][i][1])] = 0
    elif math.floor(data[0][i][0]) >= 256:
      input[255][math.floor(data[0][i][1])] = 0
    elif math.floor(data[0][i][1]) >= 256:
      input[math.floor(data[0][i][0])][255] = 0
  
  #Visualize
  if visualize == True:
    plt.imshow(input.T, cmap='gray')
    plt.gca().invert_yaxis()

  return input





def get_instances(prediction):
  #Adjust format (clusters to be 255 and rest is 0)
  prediction[prediction == 255] = 3
  prediction[prediction == 0] = 4
  prediction[prediction == 3] = 0
  prediction[prediction == 4] = 255

  #Convert to 8-bit image
  prediction = image.img_to_array(prediction, dtype='uint8')
  
  #Get 1 color channel
  cells=prediction[:,:,0]
  #Threshold
  ret1, thresh = cv2.threshold(cells, 0, 255, cv2.THRESH_BINARY)
  #Filter to remove noise
  kernel = np.ones((3,3),np.uint8)
  opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

  #Get the background
  background = cv2.dilate(opening,kernel,iterations=5)
  dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
  ret2, foreground = cv2.threshold(dist_transform,0.04*dist_transform.max(),255,0)
  foreground = np.uint8(foreground)
  unknown = cv2.subtract(background,foreground)

  #Connected Component Analysis
  ret3, markers = cv2.connectedComponents(foreground)
  markers = markers+10
  markers[unknown==255] = 0

  #Watershed
  img = cv2.merge((prediction,prediction,prediction))
  markers = cv2.watershed(img,markers)
  img[markers == -1] = [0,255,255]  

  #Get an RGB colored image
  #img2 = color.label2rgb(markers, bg_label=1)

  #Get regions
  regions = measure.regionprops(markers, intensity_image=cells)

  return regions




def draw_clusters(regions,data):
  for i in range(1,len(regions)):
    #Get the coordinates of the region
    coordinates = regions[i].coords
    #Compute the convex hull
    hull = ConvexHull(coordinates)
    #Get the indexess of the vertices
    vertices_ids = hull.vertices
    #Append real values of the vertices
    hull_vertices = []
    for j in range(0,len(vertices_ids)):
      hull_vertices.append(coordinates[vertices_ids[j]])
    #Create and plot polygon of cluster
    polygon = Polygon(hull_vertices)
    x,y = polygon.exterior.xy
    plt.plot(x,y)

  #Overlay the data points on the image
  plt.scatter(data[0][:, 0], data[0][:, 1], s=1, c='black')


def visual_clustering(data):
  input = create_input_image(data)
  result = predict_sample(input)
  regions = get_instances(result)
  draw_clusters(regions,data)