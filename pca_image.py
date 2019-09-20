import numpy as np
import pandas as pd
import cv2
from sklearn.preprocessing import StandardScaler

data = cv2.imread("3.png")
m,n,d = data.shape
resize_percent = 0.25
new_x = int(n*resize_percent)
new_y = int(m*resize_percent)
data = cv2.resize(data,(new_x,new_y),interpolation=cv2.INTER_AREA)
m,n,d = data.shape

data = np.reshape(data,(m*n,3))


sc = StandardScaler()
x_train = sc.fit_transform(data)

print(np.shape(x_train))

p = np.reshape(x_train,(m,n,3))
for channel in range(0,2):
	c = p[:,:,channel]
	c = np.absolute(c-np.mean(c))
	p[:,:,channel] = c

pc0 = p[:,:,0]
pc1 = p[:,:,1]
pc2 = p[:,:,2]



cv2.imshow('pc0',pc0)
cv2.imshow('pc1',pc1)
cv2.imshow('pc2',pc2)

cv2.waitKey(0)
cv2.destroyAllWindows()


