from scipy.misc import imread, imsave, imresize,imread,imshow
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.587*pixel[1] + 0.114*pixel[2]


    
def greyPic(image):
    grey = np.zeros((image.shape[0], image.shape[1])) # init 2D numpy array
    # get row number
    for rownum in range(len(image)):
       for colnum in range(len(image[rownum])):
          grey[rownum][colnum] = weightedAverage(image[rownum][colnum])
    return grey.astype('uint8')
    
def alphaPic(img1,img2,alpha):
    return alpha*img1+(1-alpha)*img2
    
def is_pic_file(s):
    PIC_EXTENTION='.jpg'
    if len(s)<len(PIC_EXTENTION)+1:
        return False
    if s[len(s)-len(PIC_EXTENTION):]==PIC_EXTENTION:
        return True
    return False


face_cascade = cv2.CascadeClassifier('/home/yoni/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
#eye_cascade = cv2.CascadeClassifier('/home/yoni/opencv/data/haarcascades/haarcascade_eye.xml')

files_current_dir=os.listdir('.')
pic_files_current_dir=[x for x in files_current_dir if is_pic_file(x)]


for pic_file in pic_files_current_dir:
    the_img=imread(pic_file)
    gray = cv2.cvtColor(the_img, cv2.COLOR_BGR2GRAY) #As shown in the tutorial.
    faces=face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags = cv2.CASCADE_SCALE_IMAGE)
    for (x,y,w,h) in faces:
        cv2.rectangle(the_img,(x,y),(x+w,y+h),(255,0,0),2)
    imshow(the_img)
    
