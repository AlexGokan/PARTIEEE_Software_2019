import cv2
import numpy as np
from sklearn.decomposition import PCA
import imutils


from pca_image import *
from opencvHelperFunctions import *
from old_blob_detect import *


img = cv2.imread("3.png")
m,n,d = img.shape
resize_percent = 0.5
new_x = int(n*resize_percent)
new_y = int(m*resize_percent)
img = cv2.resize(img,(new_x,new_y),interpolation=cv2.INTER_AREA)

#kmeans for color reduction below
"""
Z = img.reshape((-1,3))
Z = np.float32(Z)

K = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

showImage(res2,'reduced')
"""

orig = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray,11,17,17)
edged = cv2.Canny(gray,30,200)

contours = cv2.findContours(edged.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours,key=cv2.contourArea,reverse=True)[:10]

for c in contours:
	a = cv2.contourArea(c)
	print(a)

	
	
	
	
showImage(edged,'edged')
