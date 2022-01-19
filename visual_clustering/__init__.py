import math
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from scipy import ndimage
from matplotlib import pyplot as plt
from skimage import measure, color, io
from huggingface_hub.keras_mixin import from_pretrained_keras


class VisualClustering:
  def __init__(self, max_filter_size = 1, median_filter_size = 1):
    """self (object): containing the loaded pre-trained U-Net from the Huggingface hub
    """
    self.unet = from_pretrained_keras("tareknaous/unet-visual-clustering")
    self.max_filter_size = max_filter_size
    self.median_filter_size = median_filter_size

  def predict_sample(self, image):
    """Run inference using the U-Net model and return result

    Args:
      image (numpy.ndarray (256, 256, 1)): input image representing plotted 2D dataset

    Returns:
      result (numpy.ndarray (256, 256, 1)): predicted binary segmentation mask

    """
    prediction = self.unet.predict(image[tf.newaxis, ...])
    prediction[prediction > 0.5 ] = 1
    prediction[prediction !=1] = 0
    result = prediction[0]*255
    return result

  def create_input_image(self, data):
    #Initialize input matrix
    input = np.ones((256,256))
    #Fill matrix with data point values
    for i in range(0,len(data)):
      if math.floor(data[i][0]) < 256 and math.floor(data[i][1]) < 256:
        input[math.floor(data[i][0])][math.floor(data[i][1])] = 0
      elif math.floor(data[i][0]) >= 256:
        input[255][math.floor(data[i][1])] = 0
      elif math.floor(data[i][1]) >= 256:
        input[math.floor(data[i][0])][255] = 0

    return input

  def denoise_input(self, image):
    denoised = ndimage.median_filter(image, size=self.median_filter_size)
    return denoised

  def linear_shifting(self, data):
    max_x = max(data[:, 0])
    min_x = min(data[:, 0])
    new_max = 256
    new_min = 0

    data[:, 0] = (((data[:, 0] - min_x)*(new_max-new_min))/(max_x-min_x))+ new_min

    max_y = max(data[:, 1])
    min_y = min(data[:, 1])
    new_max_y = 256
    new_min_y = 0

    data[:, 1] = (((data[:, 1] - min_y)*(new_max_y-new_min_y))/(max_y-min_y))+ new_min_y

    return data

  def get_instances(self, prediction, data):
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

     #Obtain background
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

     #Maximum filtering
     markers = ndimage.maximum_filter(markers, size=self.max_filter_size)

     #Get regions
     regions = measure.regionprops(markers, intensity_image=cells)

     #Get Cluster IDs (Cluster Assignment)
     cluster_ids = np.zeros(len(data))

     for i in range(0,len(cluster_ids)):
       row = math.floor(data[i][0])
       column = math.floor(data[i][1])
       if row < 256 and column < 256:
         cluster_ids[i] = markers[row][column] - 10
       elif row >= 256:
         # cluster_ids[i] = markers[255][column]
         cluster_ids[i] = 0
       elif column >= 256:
         # cluster_ids[i] = markers[row][255] 
         cluster_ids[i] = 0

     cluster_ids = cluster_ids.astype('int8')
     cluster_ids[cluster_ids == -11] = 0
        
     return cluster_ids 

  def fit(self, data):
    data = self.linear_shifting(data)
    input = self.create_input_image(data)
    if self.median_filter_size == 1:
      result = self.predict_sample(input)
      labels = self.get_instances(result, data)
    else:
      denoised_input = self.denoise_input(input)
      result = self.predict_sample(denoised_input)
      labels = self.get_instances(result, data)
    return labels
