    #http://arijitgeek.me/index.php/2016/06/26/opencv-python-people-detect-in-video-feed/
	#OpenCV python HOG feature based people detector applied on video
    #Find the peopledetect.py on opencv-master/samples/python on your OpenCV installation
	
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import argparse
import imutils

from scipy.misc import imread, imsave, imresize,imread,imshow
import matplotlib.pyplot as plt
import os

def draw_detections(img, rects, thickness = 1):
	for x, y, w, h in rects:
            # the HOG detector returns slightly larger rectangles than the real objects.
            # so we slightly shrink the rectangles to get a nicer output.
            pad_w, pad_h = int(0.15*w), int(0.05*h)
            cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
			
def is_pic_file(s):
    PIC_EXTENTION='.jpg'
    if len(s)<len(PIC_EXTENTION)+1:
        return False
    if s[len(s)-len(PIC_EXTENTION):]==PIC_EXTENTION:
        return True
    return False


files_current_dir=os.listdir('.')
pic_files_current_dir=[x for x in files_current_dir if is_pic_file(x)]

hog = cv2.HOGDescriptor()
hog.setSVMDetector( cv2.HOGDescriptor_getDefaultPeopleDetector() )
for pic_file in pic_files_current_dir:
	frame=imread(pic_file)
	found,w=hog.detectMultiScale(frame, winStride=(8,8), padding=(32,32), scale=1.05)
	draw_detections(frame,found)
	imshow(frame)