from __future__ import division
from __future__ import print_function
import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

import cv2
import json,yaml
from PIL import Image
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


info = { "year":2020,"version":"1.0","description":"Coco type dataset for waymo","contributor":"Gokul","url":"http://eloop.ai","date_created":"2020"}

WAYMO_CLASSES = ['TYPE_UNKNOWN', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']
NEEDED_CLASSES = ['TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_CYCLIST']
licenses = [{"id":1,"name":"Non-commerical","url":"http://creativecommons.org/licenses/by-nc-sa/2.0"}]

images=[]
annotations=[]
imId = 1
ignoreId = 1
annotationId = 1
waymo_type= "instances"
categories = [{"id":seqId+1,"name":waymo_class,"supercategory":waymo_class}for seqId,waymo_class in enumerate(NEEDED_CLASSES)]
cat2id = {cat["name"]:catId+1 for catId,cat in enumerate(categories)}
print(categories)
print(cat2id)
PATH = '/root/waymo/data_small'
cleared = False
tf_records = os.listdir(PATH)
for indx,tf_record in enumerate(tf_records):
    print("Index is {}".format(indx))
    try:
        parent_dir_to_store = '/root/waymo/data_2d_od/segment_15'
        dataset = tf.data.TFRecordDataset(os.path.join(PATH,tf_record),compression_type='')
        for data in dataset:
            frame = open_dataset.Frame()
            if imId <500:
                path_to_save = os.path.join(parent_dir_to_store,'train')
            else:
                path_to_save = os.path.join(parent_dir_to_store,'valid')
            if imId > 500 and not cleared:
                cleared = True
                print("CLEARING")
                json_data = {"info":info,"images":images,"licenses":licenses,"type":waymo_type,"annotations":annotations,"categories":categories}

                with open(os.path.join('/root/waymo/data_2d_od/segment_15/annotations/',"train_annotations.json"),"w") as jsonfile:
                    json.dump(json_data,jsonfile,indent=4)
                images=[]
                annotations =[]
                print(images)
                print(annotations)
            frame.ParseFromString(bytearray(data.numpy()))
            for i in range(5):
                imLabels= frame.camera_labels[i].labels
                im = tf.image.decode_jpeg(frame.images[i].image).numpy()[:,:,::-1]
                if len(imLabels)==0:
                    print("IGNORING")
                    #cv2.imwrite(os.path.join(parent_dir_to_store,'ignore','{}.jpg'.format(ignoreId)),im,[int(cv2.IMWRITE_JPEG_QUALITY),40])
                    ignoreId += 1
                    continue
                cv2.imwrite(os.path.join(path_to_save,'{}.jpg'.format(imId)),im,[int(cv2.IMWRITE_JPEG_QUALITY),100])
                image = {"date_captured":"2020","file_name":str(imId)+".jpg","id":imId,"license":1,"url":"","height":im.shape[0],"width":im.shape[1]}
                images.append(image)
                imId+=1
                for label in imLabels:
                    if label.type == 0 or label.type == 3:
                        continue
                    #if label.detection_difficulty_level == 2:
                    #    cv2.imwrite(os.path.join('hard_images_segment_15','{}.jpg'.format(imId-1)),im,[int(cv2.IMWRITE_JPEG_QUALITY),100])
                    bbox = [label.box.center_x - 0.5 *label.box.length,label.box.center_y-0.5 * label.box.width,label.box.length,label.box.width]
                    annotation = {"segmentation":[],"area":label.box.length * label.box.width,"iscrowd":0,"image_id":imId-1,"bbox":bbox,"category_id":cat2id[WAYMO_CLASSES[label.type]],"id":annotationId}
                    annotations.append(annotation)
                    annotationId += 1
    except Exception as e:
        print("EXCEPTION AT {}".format(indx))
        print(e)
        pass

json_data = {"info":info,"images":images,"licenses":licenses,"type":waymo_type,"annotations":annotations,"categories":categories}

with open(os.path.join('/root/waymo/data_2d_od/segment_15/annotations/',"valid_annotations.json"),"w") as jsonfile:
    json.dump(json_data,jsonfile,indent=4)

