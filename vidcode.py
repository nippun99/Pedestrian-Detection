# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
cap=cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
while True:
	# load the image and resize it to (1) reduce detection time
	# and (2) improve detection accuracy
	ret, image=cap.read()
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()
 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
		print((float(xA)+float(xB))/2,(float(yA)+float(yB))/2)
		cordi=(float(xA)+float(xB))/2,(float(yA)+float(yB))/2
		cv2.putText(image, str(cordi), (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), lineType=cv2.LINE_AA)
	# show some information on the number of bounding boxes
	# show the output images
	
	
	cv2.imshow("After NMS", image)
	cv2.waitKey(500)
cap.release()
cv2.destroyAllWindows()		