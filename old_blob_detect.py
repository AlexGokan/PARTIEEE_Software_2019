

import cv2
import numpy as np
from sklearn.decomposition import PCA
import sklearn as skl
from matplotlib import pyplot as plt

def scale_to_image(im):
	min_val = np.min(im)
	im = im - min_val + 1
	max_val = np.max(im)
	im = im * (255/max_val)
	im = im.astype(int)
	mnd = im.shape
	out = np.zeros((mnd[0],mnd[1],3),dtype=int)
	out[:,:,0] = im
	out[:,:,1] = im
	out[:,:,2] = im
	return out
	
def create_PCA_images(imd):
	m,n,d = imd.shape
	X = np.reshape(imd,(m*n,3))

	#################################### PCA bullshit below this
	#https://docs.opencv.org/3.4.3/d1/dee/tutorial_introduction_to_pca.html
	# gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	pca = PCA(n_components=3)
	pca.fit(X)
	loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

	coeff = loadings
	# print(loadings)
	###################################### PCA bullshit above here

	Itransformed = np.matmul(X,coeff)
	final = np.zeros((m,n,3),dtype=float)
	stacked = np.zeros((m,n),dtype=float)

	for i in range(0,2):
		data = np.reshape(Itransformed[:,0],(m,n))
		# final[:,:,i] = data
		final[:,:,i] = np.absolute(data - np.mean(data))
		stacked = stacked + final[:,:,i] / np.mean(final[:,:,i])
		
	return stacked

def filter_stacked(stacked):	
	stacked_normalized = cv2.normalize(src=stacked,dst=None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
	stacked_normalized = cv2.GaussianBlur(stacked_normalized,(7,7),0)
	ret1,th1 = cv2.threshold(stacked_normalized,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return th1
	
def filter_contours(contours):
	MIN_FEATURE_SIZE__X = 50
	MAX_FEATURE_RADIUS_PX = 50
	for c in contours:
		print(c)
		#print(cv2.contourArea(c))
		#if(cv2.contourArea(c) < MIN_FEATURE_SIZE_PX):
		#	cv2.drawContours(mask,[c],-1,0,-1)
		#(x,y),radius = cv2.minEnclosingCircle(c)
		#if(radius > MAX_FEATURE_RADIUS_PX):#remove contours at the edge that are ridiculously eccentric
		#	cv2.drawContours(mask,[c],-1,0,-1)
			
	big_blobs = cv2.bitwise_and(th1,th1,mask=mask)
	contours_filtered, hierarchy_filtered = cv2.findContours(big_blobs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	return contours_filtered

def blob_detector(filename):
	
	SCALE_FACTOR = 1
	MIN_FEATURE_SIZE_PX = 50
	MAX_FEATURE_RADIUS_PX = 50#definitely play with this parameter

	xd = int(1920*SCALE_FACTOR)
	yd = int(1080*SCALE_FACTOR)
	#print(xd)
	#print(yd)

	im = cv2.imread(filename)
	#im = cv2.resize(im,(xd,yd))
	imd = np.array(im,dtype=float)

	m,n,d = imd.shape

	# X = np.zeros((m*n,3),dtype=float)

	th1 = filter_stacked(create_PCA_images(imd))#added after the fact
	
	contours, = cv2.findContours(th1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	mask = np.ones(th1.shape[:2], dtype="uint8") * 255

	print("{} contours detected".format(len(contours)))
	contours_filtered = filter_contours(contours)#added after the fact

	"""
	for c in contours_filtered:
		im_index = 0
		x,y,w,h = cv2.boundingRect(c)
		if(w<h):
			w = h
		else:
			h = w
		h = int(h * 1.25)
		w = int(w * 1.25)
		xmin = np.max([0,x-w])
		xmax = np.min([n,x+w])
		ymin = np.max([0,y-h])
		ymax = np.min([m,y+h])
		#n=image width
		#m=image height
		
		#this makes it a bounding square
		ROI = im[ymin:ymax,xmin:xmax]
		cv2.imwrite('feature'+str(im_index)+'.png',ROI)
		im_index = im_index + 1
	print('wrote '+im_index+'points of interest')
	"""