from numba import jit 
import numpy as np

@jit
def sobel_numba(img):
    Gx=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    
    rows=img.shape[0]
    cols=img.shape[1]
    
    mag=np.zeros(img.shape)
    
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            #Use numpy!
            s1= np.sum(np.multiply(Gx, np.array([[img[i-1,j-1],img[i-1,j],img[i-1,j+1]],[img[i,j-1],img[i,j],img[i,j+1]],[img[i+1,j-1],img[i+1,j],img[i+1,j+1]]])))
            s2= np.sum(np.multiply(Gy, np.array([[img[i-1,j-1],img[i-1,j],img[i-1,j+1]],[img[i,j-1],img[i,j],img[i,j+1]],[img[i+1,j-1],img[i+1,j],img[i+1,j+1]]])))
            mag[i][j]= np.sqrt(s1**2+s2**2)
    
    threshold = 70 
    for i in range(rows-1):
        for j in range(cols-1):
            #apply threshold to mag
            if mag[i,j]<=threshold:
                mag[i,j]=0
            else:
                mag[i,j]=mag[i,j]
    return mag

img_check=pickle.load(open("sobel.pickle","rb"))
img_computed=sobel_numba(img)
plt.imshow(img_computed,cmap='gray')
plt.show()


%timeit sobel_numba(img)


#Cython

%load_ext Cython
%%cython
#%%cython -a 

cimport cython
cimport numpy as c_np
from libc.math cimport round,sqrt
from cython.parallel import prange

import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)
def sobel_cython(const c_np.uint8_t[:,:] img):
    
    cdef int[3][3] Gx=[[-1,0,1],[-2,0,2],[-1,0,1]]
    cdef int[3][3] Gy=[[-1,-2,-1],[0,0,0],[1,2,1]]
        
    #Get the number of rows and columns for img
    cdef int rows=img.shape[0]
    cdef int cols=img.shape[1]
    
    cdef double[:,:] mag=np.zeros((rows,cols), dtype=np.double)
    
    
    cdef int i,j
    cdef double s1,s2
    cdef double combined_value
    
    def sobel_pixel(img,i=100,j=100):
        s1= img[i-1,j-1]*-1 + img[i-1,j]*0 + img[i-1,j+1]*1 + img[i,j-1]*-2 + img[i,j]*0 + img[i,j+1]*2 + img[i+1,j-1]*-1 + img[i+1,j]*0 + img[i+1,j+1]*1
        s2= img[i-1,j-1]*-1 + img[i-1,j]*-2 + img[i-1,j+1]*-1 + img[i, j-1]*0 + img[i,j]*0 +img[i,j+1]*0 + img[i+1,j-1]*1+ img[i+1,j]*2 +img [i+1,j+1]*1
        combined_value= np.sqrt((s1**2)+(s2**2))
        return s1,s2,combined_value

    for i in range(rows):
        for j in range(cols):
            if i==0 or j==0 or i==rows-1 or j==cols-1:
                mag[i][j]=0
            else:
                s1,s2,combined_value=sobel_pixel(img,i,j)
                mag[i][j]=combined_value       
    
    cdef int threshold = 70 #varies for application [0 255]
    for i in prange(0,rows,nogil=True):
        for j in prange(0,cols):
            if mag[i,j]<=threshold:
                mag[i,j]=0.0
            else:
                mag[i,j]=mag[i,j]
                
    return mag

img_check=pickle.load(open("sobel.pickle","rb"))
img_computed=sobel_cython(img)
plt.imshow(img_computed,cmap='gray')
plt.show()


# Benchmark/Profile
%timeit sobel_cython(img)

