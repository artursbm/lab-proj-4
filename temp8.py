import numpy as np
import cv2


cap = cv2.VideoCapture('vtes18SapoSinalizador01-2.avi')
k=0
while(1):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if (k==100):
      break
    k=k=1


cap.release()
cv2.destroyAllWindows()