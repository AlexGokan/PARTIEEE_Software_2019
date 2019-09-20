import cv2
import numpy as np

img = cv2.imread("lena.png")
img2 = cv2.imread("3.png")

m,n,d = img.shape
print([m,n,d])
resize_percent = 0.25
new_x = int(n*resize_percent)
new_y = int(m*resize_percent)
img = cv2.resize(img,(new_x,new_y),interpolation=cv2.INTER_AREA)

cv2.imshow("pic",img)

cv2.waitKey(0);
cv2.destroyAllWindows()
