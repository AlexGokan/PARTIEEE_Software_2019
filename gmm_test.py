import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn import mixture

for i in range(23):
	imnum = i
	img = cv2.imread('PARTIEEE_output_images/f01_bf/img{}.png'.format(imnum))

	for i in range(30):
		img = cv2.bilateralFilter(img,7,8,8)

	(h,w,d) = np.shape(img)

	points = []


	#green = 1, red = 2, blue = 0

	for i in range(w):
		for j in range(h):
			p1 = int(img[j,i,0])#blue
			p2 = int(img[j,i,1])#green
			p3 = int(img[j,i,2])#red
			p = [p3,p2,p1]
			points.append(p)
			#points.append(img[j,i,:])
			
	points = np.array(points)

	gmm = mixture.GaussianMixture(n_components=3,covariance_type='full')
	gmm.fit(points)

	print(gmm.means_)



	fname = "PARTIEEE_output_images/f01_colors/{}.txt".format(imnum)
	F = open(fname,"w")
	L1 = "{},{},{}\n".format(int(gmm.means_[0][0]),int(gmm.means_[0][1]),int(gmm.means_[0][2]))
	L2 = "{},{},{}\n".format(int(gmm.means_[1][0]),int(gmm.means_[1][1]),int(gmm.means_[1][2]))
	L3 = "{},{},{}\n".format(int(gmm.means_[2][0]),int(gmm.means_[2][1]),int(gmm.means_[2][2]))

	F.write(L1)
	F.write(L2)
	F.write(L3)

	F.close()

