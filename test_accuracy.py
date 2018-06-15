#coding=utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import face_recognition
import cv2
import batch_api
import time
import os
#from collections import Counter
from sklearn.svm import SVC
#from sklearn.neural_network import MLPClassifier
import Image
import numpy as np
import string
# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

#"http://172.171.52.131:8080/?action=stream"

# Get a reference to usb webcam #0 (the default one)
#video_capture = cv2.VideoCapture(0)

#/home/ben/Downloads/face_recognition-master/examples
# Initialize some variables
rootdir = r"./small_knowface/"

newf_rootdir = r"./new_knowface/"
#new_face_dir = r"./new_knowface/"
#rtsp://admin:123456bin@172.171.52.138/Streaming/Channels/1

id_name_csv = "./csv/id_name.csv"
image_csv = "./csv/image_info.csv"
know_face_csv = "./csv/know_face.csv"
rec_log_csv = "./log/"
face_log = "./face_log/"

new_name_csv = "./csv/new_name.csv"

face_locations = []
face_encodings = []
face_names = []
know_faces = {}
image_info = []

#new_face = []

#rec_log = []

id_name = {}
new_name = {}
rec_log_log = {}


know_faces = batch_api.get_know_face(know_face_csv,rootdir)
#newf_faces = batch_api.get_know_face(new_face_csv,newf_rootdir)
#new_image_info = batch_api.get_image_info(newf_info_csv,newf_rootdir)

image_info = batch_api.get_image_info(image_csv,rootdir)
id_name = batch_api.id_name_index(id_name_csv)
new_name = batch_api.new_name_index(new_name_csv)
#print(id_name)
#print(know_faces)
#print(image_info)
#obma_image= face_recognition.load_image_file("./small_knowface/1701.jpg")
#obma_face_locations = face_recognition.face_locations(obma_image)
#obma_face_encodings = face_recognition.face_encodings(obma_image)[0]
#print(obma_face_encodings)

# ovr shape
#clf = SVC()  

#ovo shape
clf = SVC(C = 10000,decision_function_shape = 'ovo')
#clf = MLPClassifier()
clf.fit(know_faces, image_info) 

cnt=0
num=float(101)

for dirpath,dirnames,filenames in os.walk(newf_rootdir):
    continue
label=new_name[8001]
print(label)
for filename in filenames:
    src = os.path.join(dirpath,filename.decode('UTF-8'))
    img = face_recognition.load_image_file(src)

    face_locations = face_recognition.face_locations(img)
    #face_encodings = face_recognition.face_encodings(img, face_locations)
    face_encodings = face_recognition.face_encodings(img)[0]

    temp = id_name[int(clf.predict([face_encodings]))]
    #label = new_name[int(string.atoi(filename[:-4],10))]
    print(int(clf.predict([face_encodings])))
    print(temp)
    print(filename)
    label = new_name[int(string.atoi(filename[:-4]))]

    if temp==label:
        cnt=cnt+1
        print(cnt)

cnt1=float(cnt)
accuracy=cnt1/num
print("该模型的准确率为：")
print(accuracy)

                    