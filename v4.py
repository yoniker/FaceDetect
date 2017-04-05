from imutils.object_detection import non_max_suppression
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2

from scipy.misc import imread, imsave, imresize,imread,imshow
import matplotlib.pyplot as plt
import os
import dlib
import sys
from skimage import io




detector = dlib.get_frontal_face_detector()



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
	img=imread(pic_file)
	dets = detector(img, 1)
	print("Number of faces detected: {}".format(len(dets)))
	for i, d in enumerate(dets):
		print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
            i, d.left(), d.top(), d.right(), d.bottom()))
		cv2.rectangle(img,(d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 2)
	imshow(img)