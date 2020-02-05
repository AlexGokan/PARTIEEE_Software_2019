"""
Ideas for further differentiation:
	otsu thresh, well separated fore and back ground?
	distribution of brightness vals

Ideas for blob detection:
	watershed algorithm
	Laplacian of gaussian / difference of gaussian
	https://en.wikipedia.org/wiki/Blob_detection
"""


import cv2
import numpy as np


filepath = "targets_01_widecrop.png"
dim = (1920//2,1080//2)

img = cv2.imread(filepath,0)
vis = cv2.imread(filepath,1)
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
vis = cv2.resize(vis, dim, interpolation = cv2.INTER_AREA)



mser = cv2.MSER_create()

regions, _ = mser.detectRegions(img)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

visc = vis.copy()
vism = vis.copy()

cv2.polylines(visc, hulls, 1, (0, 255, 0))

"""
cv2.imshow('img', vis)

cv2.waitKey(0)
"""

mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

for contour in hulls:

    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

#this is used to find only text regions, remaining are ignored
text_only = cv2.bitwise_and(img, img, mask=mask)
retval,t = cv2.threshold(text_only,1,255,cv2.THRESH_BINARY)

contours,hierarchy = cv2.findContours(t,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

inertia = [cv2.contourArea(c)/cv2.arcLength(c,True) for c in contours]

idx = 0



matched = []

inertia_thresh = 0.2
size_ub = 500
size_lb = 40

for c in contours:
	
	sz = cv2.contourArea(c)
	perim = cv2.arcLength(c,True)
	inertia = perim/sz
	
	
	mask = np.zeros(img.shape,np.uint8)
	cv2.drawContours(mask,[c],0,255,-1)
	pixelpoints = np.transpose(np.nonzero(mask))

#	cv2.imshow("mask",mask)
#	cv2.waitKey(0)

	only_roi = cv2.bitwise_and(mask,img)
	
#	print(pixelpoints)
	
	roi_raw = []
	for i in range(0,np.shape(pixelpoints)[0]):
		roi_raw.append(img[pixelpoints[i][0],pixelpoints[i][1]])
		
	deviation = np.std(roi_raw)
	print("{} {} ".format(idx,deviation))

	

	"""
	print(max_in_roi)
	print(min_in_roi)
	print("............")
	"""
	
	#print(sz)
	
	if(inertia > 0.2 and sz > size_lb and sz < size_ub):
		matched.append(c)
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	#visc = cv2.putText(visc,"r{0:.3f}".format(roundness),(cX,cY-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	#visc = cv2.putText(visc,"a{}".format(sz),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	visc = cv2.putText(visc,"{}".format(idx),(cX-20,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	visc = cv2.putText(visc,"{0:.3f}".format(deviation),(cX+20,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	#visc = cv2.putText(visc,"p{0:.4f}".format(perim),(cX,cY+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	idx = idx + 1
	
cv2.imshow('all MSERs',visc)


print("-----------------")
print(len(contours))
print(len(matched))


idx = 0

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in matched]
cv2.polylines(vism, hulls, 1, (0, 255, 0))


for c in matched:

	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	vism = cv2.putText(vism,"{}".format(idx),(cX+10,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	idx = idx + 1

#cv2.imshow("text only", t)

cv2.imshow('matched by heuristics',vism)



cv2.waitKey(0)

cv2.destroyAllWindows()
