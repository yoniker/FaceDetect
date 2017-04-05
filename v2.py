# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

from scipy.misc import imread, imsave, imresize,imread,imshow
import matplotlib.pyplot as plt
import os


"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())
"""


# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def is_pic_file(s):
    PIC_EXTENTION='.jpg'
    if len(s)<len(PIC_EXTENTION)+1:
        return False
    if s[len(s)-len(PIC_EXTENTION):]==PIC_EXTENTION:
        return True
    return False



files_current_dir=os.listdir('.')
pic_files_current_dir=[x for x in files_current_dir if is_pic_file(x)]


for pic_file in pic_files_current_dir:
	image=imread(pic_file)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()
 
	# detect people in the image
	(rects, weights) = hog.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.05)
 
	# draw the original bounding boxes
	for (x, y, w, h) in rects:
		cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
 
	# apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
	rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
	pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
 
	# draw the final bounding boxes
	for (xA, yA, xB, yB) in pick:
		cv2.rectangle(orig, (xA, yA), (xB, yB), (0, 255, 0), 2)
	imshow(orig)