import cv2
import numpy as np
from sklearn.decomposition import PCA
import imutils

img = cv2.imread("targets_01.png")
m,n,d = img.shape

img = img[912:1963,1263:2343,:]
m,n,d = img.shape

img = cv2.resize(img,(int(m/3),int(n/3)))
m,n,d = img.shape

S = img * 0;
for i in range(1,m-1):
	for j in range(1,n-1):
		px = img[i,j,:]
		tl = img[i-1,j-1,:]
		tr = img[i-1,j+1,:]
		bl = img[i+1,j-1,:]
		br = img[i+1,j+1,:]
		m = [0,0,0]
		m[0] = np.mean([tl[0],tr[0],bl[0],br[0]])
		m[1] = np.mean([tl[1],tr[1],bl[1],br[1]])
		m[2] = np.mean([tl[2],tr[2],bl[2],br[2]])
		dist =((px[0]-m[0])**2) + ((px[1]-m[1])**2) + ((px[2]-m[2])**2)
		dist = int(np.sqrt(dist))
		S[i,j,0] = dist
		S[i,j,1] = dist
		S[i,j,2] = dist
		"""
		for ch in range(0,3):
			px = img[i,j,ch]
			tl = img[i-1,j-1,ch]
			tr = img[i-1,j+1,ch]
			bl = img[i+1,j-1,ch]
			br = img[i+1,j+1,ch]
			
			m = np.mean([tl,tr,bl,br])
			diff = float(px) - m
			S[i,j,ch] = int(diff)
		"""




#S = cv2.resize(S,(int(m*8),int(m*8)))
cv2.imshow("img",S)

cv2.waitKey(0)

cv2.destroyAllWindows()

