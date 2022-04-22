import numpy as np

def sobel_numpy(img):
    Gx=np.array([-1,0,1,-2,0,2,-1,0,1],dtype = int)
    Gy=np.array([-1,-2,-1,0,0,0,1,2,1],dtype = int)
    
    rows=img.shape[0]
    cols=img.shape[1]
    
    mag=np.zeros(img.shape)
    
    for i in range(1,rows-1):
        for j in range(1,cols-1):
            #Use numpy!
            s1= np.dot(Gx, np.array([img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]],dtype = int))
            s2= np.dot(Gy, np.array([img[i-1,j-1],img[i-1,j],img[i-1,j+1],img[i,j-1],img[i,j],img[i,j+1],img[i+1,j-1],img[i+1,j],img[i+1,j+1]],dtype = int))
            mag[i][j]= np.sqrt(s1**2+s2**2)
    
    threshold=70
    #apply threshold. Hint: Look at np.vectorize
    for i in range(rows-1):
        for j in range(cols-1):
            #apply threshold to mag
            if mag[i,j]<=threshold:
                mag[i,j]=0
            else:
                mag[i,j]=mag[i,j]
   
    return mag

img_check=pickle.load(open("sobel.pickle","rb"))
img_computed=sobel_numpy(img)
plt.imshow(img_computed,cmap='gray')
plt.show()


#benchmark/profile
%timeit sobel_numpy(img)
