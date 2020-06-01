from __future__ import print_function
from __future__ import division
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
import glob
import os
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument('split', choices=['training','validation'])
parser.add_argument('--out-dir',default='./tmp')
args = parser.parse_args()

info = { "year":2020,"version":"1.0","description":"Coco type dataset for waymo","contributor":"Gokul","url":"http://eloop.ai","date_created":"2020"}

WAYMO_CLASSES = ['TYPE_UNKNOWN', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']
NEEDED_CLASSES = ['TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_CYCLIST']
licenses = [{"id":1,"name":"Non-commerical","url":"http://creativecommons.org/licenses/by-nc-sa/2.0"}]

images=[]
annotations=[]
imId = 207000 
ignoreId = 1
annotationId = 2000000
waymo_type= "instances"
categories = [{"id":seqId+1,"name":waymo_class,"supercategory":waymo_class}for seqId,waymo_class in enumerate(NEEDED_CLASSES)]
cat2id = {cat["name"]:catId+1 for catId,cat in enumerate(categories)}

url_template = 'gs://waymo_open_dataset_v_1_0_0/{split}/{split}_%04d.tar'.format(split=args.split)
if args.split == 'training':
    num_segs = 32
elif args.split == 'validation':
    num_segs = 8

save_parent_dir = '/root/waymo/dataset'

annotation_path = '/root/waymo/dataset/annotations'

if args.split == 'training':
    img_dir = os.path.join(save_parent_dir,'train')
    annotation_file = os.path.join(annotation_path,'train_annotations.json')
else:
    img_dir = os.path.join(save_parent_dir,'valid')
    annotation_file = os.path.join(annotation_path,'valid_annotations.json')

    
if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

for seg_id in range(0,num_segs):
    flag = os.system('gsutil cp '+ url_template %seg_id + ' ' + args.out_dir)
    assert flag == 0 ,'Failed to download segment %d. Make sure gsuil is installed'%seg_id
    os.system('cd %s; tar xf %s_%04d.tar'%(args.out_dir,args.split,seg_id))
    tfrecords = sorted(glob.glob('%s/*.tfrecord'%args.out_dir))
    for record in tfrecords:
        dataset = tf.data.TFRecordDataset(record,compression_type='')
        for data in dataset:
            only_lidar = False
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            quality= 40
            if 'Night' in frame.context.stats.time_of_day:
                quality = 100
            if 'sunny' not in frame.context.stats.weather:
                quality = 100
            for i in range(5):
                random_num = random.random()
                if i == 0 and random_num > 0.5:
                    continue
                if i > 0 and random_num > 0.25:
                    continue
                labels = frame.camera_labels
                if not labels:
                    labels = frame.projected_lidar_labels
                    only_lidar = True
                if len(labels) == 0:
                    break
                if i>0 and only_lidar:
                    continue
                imLabels =labels[i].labels
                im = tf.image.decode_jpeg(frame.images[i].image).numpy()[:,:,::-1]
                if len(imLabels) == 0:
                    ignoreId += 1
                    continue
                cv2.imwrite(os.path.join(img_dir,'{}.jpg'.format(imId)),im,[int(cv2.IMWRITE_JPEG_QUALITY),quality])
                
                image = {"date_captured":"2020","file_name":str(imId)+".jpg","id":imId,"license":1,"url":"","height":im.shape[0],"width":im.shape[1]}
                images.append(image)
                print(imId)
                imId+=1
                for label in imLabels:
                    if label.type == 0 or label.type == 3:
                        continue

                    bbox = [label.box.center_x - 0.5 *label.box.length,label.box.center_y-0.5 * label.box.width,label.box.length,label.box.width]
                    annotation = {"segmentation":[],"area":label.box.length * label.box.width,"iscrowd":0,"image_id":imId-1,"bbox":bbox,"category_id":cat2id[WAYMO_CLASSES[label.type]],"id":annotationId}
                    annotations.append(annotation)
                    annotationId += 1
        print("Removing Record")
        os.remove(record)
    print("Removing entire segment")
    os.system('cd %s; rm %s_%04d.tar'%(args.out_dir,args.split,seg_id))


json_data = {"info":info,"images":images,"licenses":licenses,"type":waymo_type,"annotations":annotations,"categories":categories}

with open(annotation_file,"w") as jsonfile:
    json.dump(json_data,jsonfile,indent=4)
