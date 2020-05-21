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

FILENAME = '/root/waymo/data_small/segment-14165166478774180053_1786_000_1806_000_with_camera_labels.tfrecord'

dataset = tf.data.TFRecordDataset(FILENAME,compression_type='')
count = 0
frame_count = 0
for data in dataset:
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    for i in range(5):
        im = tf.image.decode_jpeg(frame.images[i].image).numpy()[:,:,::-1]
        for j in range(10):
            cv2.imwrite(os.path.join('image_sample','quality_{}_{}.jpg'.format(j,i)),im,[int(cv2.IMWRITE_JPEG_QUALITY),j*10])
        im = cv2.resize(im,(1280,1280))
        cv2.imwrite(os.path.join('image_sample','resize_{}.jpg'.format(i)),im)
    break
