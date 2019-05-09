import cv2
import numpy as np
from matplotlib import pyplot as plt
import time

captura =cv2.VideoCapture(0)

k=0;
while(1):

    ret1, img = captura.read()
    k=k+1;
    #img = cv2.imread('color_0_0002.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b,g,r = cv2.split(img)    
    tb = cv2.inRange(b, 0, 120)
            
    #gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh1 = cv2.inRange(gray, 45,150)
    thresh1 = cv2.erode(thresh1, None, iterations=2)
    thresh1 = cv2.dilate(thresh1, None, iterations=2)
    thresh1=cv2.multiply(1/255,thresh1)
    result1=cv2.multiply(gray,thresh1)
    cv2.imshow("Image",result1)
    m=cv2.waitKey(1) 
    if (m==ord('q')): 
        break
    
    if (k==1000):
        break

thresh1=cv2.multiply(1/255,thresh1)
result1=cv2.multiply(gray,thresh1)


#plt.imshow(tb,'gray')

plt.xticks([]),plt.yticks([])

cv2.imshow("Image",result1)
cv2.waitKey(0)
cv2.destroyWindow("Image")
#plt.show()
# cv2.imwrite("thr.png",thresh1)

captura.release()
cv2.destroyAllWindows()