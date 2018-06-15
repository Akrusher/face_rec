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
#import Image
import numpy as np
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

#newf_rootdir = r"./new_knowface/"
#new_face_dir = r"./new_knowface/"
#rtsp://admin:123456bin@172.171.52.138/Streaming/Channels/1

id_name_csv = "./csv/id_name.csv"
image_csv = "./csv/image_info.csv"
know_face_csv = "./csv/know_face.csv"
rec_log_csv = "./log/"
face_log = "./face_log/"

#new_face_csv = "./new_face.csv"
#newf_info_csv = "./newf_info.csv" 

face_locations = []
face_encodings = []
face_names = []
know_faces = {}
image_info = []

#new_face = []

#rec_log = []

id_name = {}
rec_log_log = {}


know_faces = batch_api.get_know_face(know_face_csv,rootdir)
#newf_faces = batch_api.get_know_face(new_face_csv,newf_rootdir)
#new_image_info = batch_api.get_image_info(newf_info_csv,newf_rootdir)

image_info = batch_api.get_image_info(image_csv,rootdir)
id_name = batch_api.id_name_index(id_name_csv)
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

process_this_frame = True


# clear a index of name to id ,use to register 
#rtsp://admin:123456bin@172.171.52.138/Streaming/Channels/1
print("GO")

#rec_time = time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time()))
#print(time.localtime(time.time()).tm_year)

#video_capture = cv2.VideoCapture("http://172.171.52.125:8080/?action=stream")
#video_capture = cv2.VideoCapture("http://172.171.52.132:8080/?action=stream")
video_capture = cv2.VideoCapture(0)
from wxpy import *
bot= Bot(console_qr=False,cache_path=True)

my_friend = bot.friends().search(u'成汉林',sex=MALE)[0]
#my_friend.send("这是一条测试消息，收到请勿回复")
'''item={
1964:1,
3202:1,
50500:1,
50501:1,
50502:1,
50503:1,
50504:1,
50505:1,
50506:1,
50507:1,
50508:1,
50509:1,
50510:1,
50511:1,
50512:1,
50513:1,
50514:1,
50515:1,
50516:1,
50517:1,
50518:1,
50519:1,
7020020:1,
}'''
current_ID=0
last_ID=0
temp = "Unknown"
while True:
    
    # Grab a single frame of video 172.171.52.115
    #video_capture = cv2.VideoCapture("http://192.168.1.1:8080/?action=stream")
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    if False == ret:
        video_capture.release()
        print("webcam can not capture"+time.strftime('%Y-%m-%d-%H:%M:%S',time.localtime(time.time())))
        
    else:
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        #print(small_frame)
        #print(type(small_frame))
        #array_frame=np.array(small_frame)
        #print(type(array_frame))
        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)
            name = "Please Register"
            rec_time = "Unknown"
            ID = "Unknown"
            
            face_names = []
            #print("nihao")
            #print(id_name[1001])
            '''for face_encoding in face_encodings:
                print(int(clf.predict([face_encoding])))'''
                #print(id_name[int(clf.predict([face_encoding]))])          
                
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                #if int(clf.predict([face_encoding])) in ids:
                try:
                    temp = id_name[int(clf.predict([face_encoding]))]
                    #print("test1")
                except:
                    name = "Please Register"
                    #print("test2")
                else:
                    #print("test3")
                    ID = str(int(clf.predict([face_encoding])))

                    current_ID = int(clf.predict([face_encoding]))
                    
                    #print(clf.score(know_faces[image_info[clf.predict([face_encoding]))],clf.predict([face_encoding])))
                    name = temp + ID
                    time_struct = time.localtime(time.time())
                    #print(time_struct)

                    rec_time = time.strftime("%Y-%m-%d-%H:%M:%S",time_struct)
                    #print(rec_time)
                    rec_date = str(time_struct.tm_year)+'-'+str(time_struct.tm_mon)+'-'+str(time_struct.tm_mday)
                    #print(rec_date)
                    src_st = face_log + rec_date
                    #print(src_st)
                    if not os.path.exists(src_st):
                        os.mkdir(src_st)
                        cv2.imwrite(src_st+'/'+rec_time+name+'.jpg',frame)
                    else:
                        cv2.imwrite(src_st+'/'+rec_time+name+'.jpg',frame)

                    batch_api.store_rec_log(name,ID,rec_time,rec_date,rec_log_log,rec_log_csv)

                    #cv2.imwrite('./face_log/'+rec_time+'.jpg',frame)
                """
                ## distance cal,simple but un-accurate

                match = face_recognition.compare_faces(know_faces, face_encoding,0.44)
                name = "Unknown"

                #print(clf.score([face_encoding],clf.predict([face_encoding])))
                """
               
                face_names.append(name)

       
        process_this_frame = not process_this_frame

        
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 1
            right *= 1
            bottom *= 1
            left *= 1


            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom -30), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(frame, ID, (left + 6, bottom - 6), font, 0.8, (0, 255, 0), 1)
            if ((current_ID>=100001)and(current_ID<=101001)):
                name= "Please Register"
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.8, (0, 255, 0), 1)

        
        # Display the resulting image
        
        cv2.imshow('Video', frame)
        if ((current_ID<100001)or(current_ID>101001)):
            if last_ID!=current_ID:
                my_friend.send("检测到%s出现"%temp)
                cv2.imwrite('1.jpg',frame)
                my_friend.send_image('1.jpg')
                last_ID=current_ID             
                os.remove("1.jpg")
        elif ((current_ID>=100001)or(current_ID<=101001)):
            if last_ID!=current_ID:
                my_friend.send("You are new Person，Welcome my friend！")
                cv2.imwrite('1.jpg',frame)
                my_friend.send_image('1.jpg')
                last_ID=current_ID             
                os.remove("1.jpg")
        if ((name == "Please Register")):
            if last_ID!=current_ID:
                my_friend.send("You are new Person，Welcome my friend！")
                cv2.imwrite('1.jpg',frame)
                my_friend.send_image('1.jpg')
                last_ID=current_ID             
                os.remove("1.jpg")
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
