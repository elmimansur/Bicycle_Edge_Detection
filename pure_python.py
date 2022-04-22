import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

import numpy as np
import pickle

img=mpimg.imread('Bikesgray.jpg')
plt.imshow(img,cmap='gray')
plt.show()


def sobel_pixel(img,i=100,j=100):
    s1= img[i-1,j-1]*-1 + img[i-1,j]*0 + img[i-1,j+1]*1 + img[i,j-1]*-2 + img[i,j]*0 + img[i,j+1]*2 + img[i+1,j-1]*-1 + img[i+1,j]*0 + img[i+1,j+1]*1
    s2= img[i-1,j-1]*-1 + img[i-1,j]*-2 + img[i-1,j+1]*-1 + img[i, j-1]*0 + img[i,j]*0 +img[i,j+1]*0 + img[i+1,j-1]*1+ img[i+1,j]*2 +img [i+1,j+1]*1
    combined_value= np.sqrt((s1**2)+(s2**2))
    
    return s1,s2,combined_value
    
s1,s2,combined=sobel_pixel(img)



#create own sobel
def sobel_py(img):
    
    rows=img.shape[0]
    cols=img.shape[1]
    
    mag=np.zeros(img.shape) 
    
    for i in range(rows):
        for j in range(cols):
            if i==0 or j==0 or i==rows-1 or j==cols-1:
                mag[i][j]=0
            else:
                combined_value=sobel_pixel(img,i,j)[2]
                mag[i][j]=combined_value
            
    threshold=70
    for i in range(rows):
        for j in range(cols):
            #apply threshold to mag
            if mag[i,j]<=threshold:
                mag[i,j]=0
            else:
                mag[i,j]=mag[i,j]
                
    
    return mag

img_check=pickle.load(open("sobel.pickle","rb"))
img_computed=sobel_py(img)
plt.imshow(img_computed,cmap='gray')
plt.show()


#benchmark / profile
%timeit sobel_py(img)
%prun sobel_py(img)
%load_ext line_profiler 
%lprun -f sobel_py sobel_py(img) 
#memory
%load_ext memory_profiler
%mprun -f sobel_py sobel_py(img)
