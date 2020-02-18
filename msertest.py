"""
Ideas for further differentiation:
	otsu thresh, well separated fore and back ground?
	distribution of brightness vals

Ideas for blob detection:
	watershed algorithm
	Laplacian of gaussian / difference of gaussian
	https://en.wikipedia.org/wiki/Blob_detection
"""

"""
TODO: 
determine MSER upper and lower limits for size
check cv2 version, 4.0 vs 4.1 for mser image size
"""

import cv2
import numpy as np
import os
import sys

print(cv2.__version__)

input_image_path = sys.argv[1]
output_path = sys.argv[2]

filepath = "targets_01_widecrop.png"
#filepath = "3.png"
filepath = input_image_path

dim = (1920//2,1080//2)

img = cv2.imread(filepath,0)
vis = cv2.imread(filepath,1)

print(img.shape)
#img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
#vis = cv2.resize(vis, dim, interpolation = cv2.INTER_AREA)



mser = cv2.MSER_create()

regions, _ = mser.detectRegions(img)

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

visc = vis.copy()
vism = vis.copy()

img_for_crop = vis.copy()

cv2.polylines(visc, hulls, 1, (0, 255, 0))


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

inertia_thresh = 0.1
size_lb = int(img.shape[1]*img.shape[0]*0.000075)

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
	print("{} {} {}".format(idx,deviation,sz))


	
	#print(sz)
	
	if(inertia > inertia_thresh and sz > size_lb):
		matched.append(c)
	M = cv2.moments(c)
	cX = int(M["m10"] / M["m00"])
	cY = int(M["m01"] / M["m00"])
	
	x,y,w,h = cv2.boundingRect(c)
	#visc = cv2.putText(visc,"r{0:.3f}".format(roundness),(cX,cY-20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	#visc = cv2.putText(visc,"a{}".format(sz),(cX,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	#visc = cv2.putText(visc,"{}".format(idx),(cX-20,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	#visc = cv2.putText(visc,"{0:.3f}".format(inertia),(cX+20,cY),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	#visc = cv2.putText(visc,"{0:.3f}".format(sz),(cX+40,cY+10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
	#visc = cv2.putText(visc,"p{0:.4f}".format(perim),(cX,cY+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	visc = cv2.putText(visc,"p{0:.4f}".format(x),(cX+20,cY),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	visc = cv2.putText(visc,"p{0:.4f}".format(y),(cX+40,cY+20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
	
	
	idx = idx + 1
	
	
visc = cv2.resize(visc,dim,interpolation = cv2.INTER_AREA)
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
	
dir_path = "PARTIEEE_output_images/{}".format(output_path)
print(dir_path)
if(os.path.isdir(dir_path)):
	print("dir exists!")
else:
	os.mkdir(dir_path)
	print("dir does not exist")

	
idx = 0
padding = 2.0
ylim,xlim,depth = img_for_crop.shape
for c in matched:
	x,y,w,h = cv2.boundingRect(c)
	
	lside = x - int(padding*w/2) + int(w/2)
	rside = x + int(padding*w/2) + int(w/2)
	upside = y - int(padding*h/2) + int(h/2)
	downside = y + int(padding*h/2) + int(h/2)

	if(upside < 0):
		upside = 0
		
	if(lside < 0):
		lside = 0
		
	if(downside >= ylim):
		downside = ylim - 1
	if(rside >= xlim):
		rside = xlim - 1
	
	
	
	print("{} {} {} {}".format(x,y,w,h))
	print("{} {}, {} {}".format(lside,rside,upside,downside))
	
	
	print("--------------------------------------------")
	
	#cropped = img_for_crop[lside:rside][upside:downside]
	#cropped = img_for_crop[0:100,0:100].copy()
	cropped = img_for_crop[upside:downside,lside:rside].copy()
	print("{},{}".format(rside-lside,downside-upside))
	print(cropped.shape)
	cropped = cv2.resize(cropped,(cropped.shape[1]*4,cropped.shape[0]*4),interpolation = cv2.INTER_AREA)
	
	print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
	cv2.imshow("cropped{}".format(idx),cropped)
	
	cv2.imwrite("{}/img{}.png".format(dir_path,idx),cropped)
	
	idx = idx + 1




vism = cv2.resize(vism,dim,interpolation = cv2.INTER_AREA)
cv2.imshow('matched by heuristics',vism)





cv2.waitKey(0)

cv2.destroyAllWindows()
