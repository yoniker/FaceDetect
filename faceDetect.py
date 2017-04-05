from scipy.misc import imread, imsave, imresize,imread,imshow
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

    
def is_pic_file(s):
    PIC_EXTENTION='.jpg'
    if len(s)<len(PIC_EXTENTION)+1:
        return False
    if s[len(s)-len(PIC_EXTENTION):]==PIC_EXTENTION:
        return True
    return False


face_cascade = cv2.CascadeClassifier('C:\Users\Yoni\Downloads\opencv\opencv\sources\data\haarcascades\haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('C:\Users\Yoni\Downloads\opencv\opencv\sources\data\haarcascades\haarcascade_eye.xml')

files_current_dir=os.listdir('.')
pic_files_current_dir=[x for x in files_current_dir if is_pic_file(x)]


for pic_file in pic_files_current_dir:
    the_img=imread(pic_file)
    #gray = cv2.cvtColor(the_img, cv2.COLOR_BGR2GRAY) #As shown in the tutorial.
    gray=the_img
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    rect_img=the_img
    for (x,y,w,h) in faces:
        rect_img = cv2.rectangle(rect_img,(x,y),(x+w,y+h),(255,0,0),2)
    imshow(rect_img)
    