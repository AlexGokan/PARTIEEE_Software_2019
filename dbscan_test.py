from sklearn.cluster import DBSCAN
import numpy as np
import cv2

img = cv2.imread("targets_01.png")
m,n,d = img.shape

img = img[912:1963,1263:2343,:]
m,n,d = img.shape
img = cv2.resize(img,(int(m/4),int(n/4)))
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


m = np.shape(img)[0]
n = np.shape(img)[1]

yy,xx = np.meshgrid(np.arange(0,n,1),np.arange(0,m,1))

print(np.shape(xx))
print(np.shape(img))

L = np.reshape(img,(-1,1))
yy = np.reshape(yy,(-1,1))
xx = np.reshape(xx,(-1,1))

D = np.zeros((m*n,3))

for i in range(m*n):
	D[i,0] = xx[i]
	D[i,1] = yy[i]
	D[i,2] = L[i]
	
clustering = DBSCAN(eps=4,min_samples=2).fit(D)

print(clustering.labels_)

