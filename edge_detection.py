import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import numpy as np
import pickle


#for one pixel
img=mpimg.imread('Bikesgray.jpg')
plt.imshow(img,cmap='gray')
plt.show()

def sobel_pixel(img,i=100,j=100):
    s1= img[i-1,j-1]*-1 + img[i-1,j]*0 + img[i-1,j+1]*1 + img[i,j-1]*-2 + img[i,j]*0 + img[i,j+1]*2 + img[i+1,j-1]*-1 + img[i+1,j]*0 + img[i+1,j+1]*1
    s2= img[i-1,j-1]*-1 + img[i-1,j]*-2 + img[i-1,j+1]*-1 + img[i, j-1]*0 + img[i,j]*0 +img[i,j+1]*0 + img[i+1,j-1]*1+ img[i+1,j]*2 +img [i+1,j+1]*1
    combined_value= np.sqrt((s1**2)+(s2**2))
    
    return s1,s2,combined_value
  
s1,s2,combined=sobel_pixel(img)
