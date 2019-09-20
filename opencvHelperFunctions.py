import cv2
def showImage(im,title):
	cv2.imshow(title,im)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
