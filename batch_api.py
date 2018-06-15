# -*- coding: utf-8 -*-
import numpy as np
import face_recognition
import os
import os.path
import string
import csv

import matplotlib.pyplot as plt
import sys
import sklearn
import sklearn.metrics.pairwise as pw

#caffe_root = '/home/akrusher/caffe'  
#sys.path.insert(0, caffe_root + 'python')
'''import caffe

feat = []

def initilize():
    print 'model initilizing...'
    deployPrototxt = "./VGG_FACE_deploy.prototxt"
    modelFile = "./VGG_FACE.caffemodel"
    caffe.set_mode_cpu()
    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net
net = initilize()
def extractFeature(image):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) # create transformer for the input called 'data'

    transformer.set_transpose('data', (2,0,1)) # move image channls to outermost dimension
    #transformer.set_mean('data', np.load(caffe_root + meanFile).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # rescale from [0,1] to [0,255]
    transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR
    # set net to batch size of 1 
    net.blobs['data'].reshape(1, 3,224, 224)  

    feature = []
    img = caffe.io.load_image(image)
    #print image, img.shape
    img = caffe.io.resize_image(img, (224,224))
    #print image, img.shape
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    #print net.blobs['data'].data.shape
    out = net.forward()
    #featureleft = np.float64(out['fc7'])
    #print featureleft.shape
    #feature = np.float64(net.blobs['fc7'].data)
    #print feature, type(feature)
    #print feature
    return out #list'''
"""
def get_know_face():
	# if the image_code is available then load it into the face_image
	know_face_csv = "./know_face.csv"
	know_faces = np.loadtxt(open(know_face_csv,"rb"),delimiter = ',',skiprows = 0)
	return know_faces


	with open(know_face_json) as know_face_obj:
		know_faces = json.load(know_face_obj)
		
	return know_faces


def store_know_face(know_faces):
	know_face_csv = "./know_face.csv"
	np.savetxt(know_face_csv,know_faces,delimiter = ',')

	
	with open(know_face_json,'a') as know_face_obj:
		flag = 0
		while flag < len(know_faces):
			know_face_obj.write(np.array_str(know_faces[flag]))
			flag = flag + 1
"""
'''def extractFeature1(image):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape}) # create transformer for the input called 'data'

    transformer.set_transpose('data', (2,0,1)) # move image channls to outermost dimension
    #transformer.set_mean('data', np.load(caffe_root + meanFile).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  # rescale from [0,1] to [0,255]
    transformer.set_channel_swap('data', (2,1,0)) # swap channels from RGB to BGR
    # set net to batch size of 1 
    net.blobs['data'].reshape(1, 3,224, 224) 

    feature = []
    #img = caffe.io.imread(image)
    #print image, img.shape
    img = caffe.io.resize_image(image, (224,224))
    #print image, img.shape
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    #print net.blobs['data'].data.shape
    out = net.forward()
    #featureleft = np.float64(out['fc7'])
    #print featureleft.shape
    #feature = np.float64(net.blobs['fc7'].data)
    #print feature, type(feature)
    #print feature
    return out #list
def face_encodings_cnn(face_image):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """ 
    feat = extractFeature(face_image)
    feature = np.array(feat['prob'])
    return [np.array(feature)]
def face_encodings_cnn_rec(face_image1):
    """
    Given an image, return the 128-dimension face encoding for each face in the image.

    :param face_image: The image that contains one or more faces
    :param known_face_locations: Optional - the bounding boxes of each face if you already know them.
    :param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
    """ 
    feat = extractFeature1(face_image1)
    feature = np.array(feat['prob'])
    return [np.array(feature)]'''

def know_face_encoding(rootdir):
    know_faces = []
    filenames = []
    for dirpath,dirnames,filenames in os.walk(rootdir): 
        continue
    cnt=1
    for filename in filenames:
        src = os.path.join(dirpath,filename.decode('UTF-8'))
        print(src)
        #print(cnt)
        #print(type(face_recognition.face_encodings(face_recognition.load_image_file(src))[0]))
        know_faces.append(face_recognition.face_encodings(face_recognition.load_image_file(src))[0])
        cnt=cnt+1
        print(src)
        #know_faces.append(face_recognition.face_encodings(filename,face_locations))
    return know_faces


def walk_image(rootdir):
	image_info = []
	filenames = []
	for dirpath,dirnames,filenames in os.walk(rootdir):
		continue

	for filename in filenames:
		image_info.append(string.atoi(filename[:-4]))
	
	return image_info


def store_know_face(know_faces,know_face_csv):
    np.savetxt(know_face_csv,know_faces,delimiter = ',')


def store_image_info(image_info,image_csv):
	np.savetxt(image_csv,image_info,delimiter = ',')


def get_image_info(image_csv,rootdir):
	#the index of image info construct
	try:
		with open(image_csv,"rb") as image_info_obj:
			image_info = np.loadtxt(image_info_obj,delimiter = ',',skiprows = 0)
	except:
		print ("image_csv not exist")
		image_info = walk_image(rootdir)
		store_image_info(image_info,image_csv)
	else:
		return image_info


def get_know_face(know_face_csv,rootdir):
    # if the image_code is available then load it into the face_image, otherwise,store the face
    try:
        with open(know_face_csv,"rb") as know_face_obj:
            know_faces = np.loadtxt(know_face_obj,delimiter = ',',skiprows = 0)
    except:
    	print ("know_face_csv not exist")
        know_faces = know_face_encoding(rootdir)
        store_know_face(know_faces,know_face_csv)
    else:
        return know_faces


def id_name_index(id_name_csv):
	id_name = {}
	try:
		with open(id_name_csv,"rb") as id_name_obj:
			reader =csv.reader(id_name_obj)
			header_row = next(reader)
			for row in reader:
				id_name[int(row[0])] = row[1]
	
	except:
		print("id_name_csv not exist")
	else:
		return id_name

def new_name_index(new_name_csv):
    new_name = {}
    try:
        with open(new_name_csv,"rb") as new_name_obj:
            reader =csv.reader(new_name_obj)
            header_row = next(reader)
            for row in reader:
                new_name[int(row[0])] = row[1]
    
    except:
        print("new_name_csv not exist")
    else:
        return new_name

def store_rec_log(name,ID,rec_time,rec_date,rec_log_log,rec_log_csv):
	src_log = rec_log_csv+rec_date+'.csv'
	try:
		with open(src_log,"ab") as rec_log_obj:
			writer = csv.writer(rec_log_obj)
			#print(rec_log_log.keys())
			if ID not in rec_log_log.keys():
				writer.writerow([name]+[rec_time])
				rec_log_log[ID] = '1'
			elif rec_log_log[ID] != '1':
				writer.writerow([name]+[rec_time])	
	except:
		print("open rec_log_csv error")

"""
def rec_log(name,ID):
	rec_log = []

def get_rec_log(rec_log_csv):
	try:
		with open(rec_log_csv,"rb") as rec_log_obj:
			rec_log = np.loadtxt(rec_log_obj,delimiter = ',',skiprows = 0)
    except:
    	print("know_face_csv not exist")
    	know_faces = know_face_encoding(rootdir)
    	store_know_face(know_faces,know_face_csv)
    else:
    	return know_faces

def get_new_face():
"""