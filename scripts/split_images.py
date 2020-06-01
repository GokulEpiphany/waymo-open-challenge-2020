import os
import cv2
from pycocotools.coco import COCO
image_path = '/root/waymo/data_2d_od/segment_15/train/47.jpg'
annotations = '/root/waymo/data_2d_od/segment_15/annotations/train_annotations.json'

coco_anns = COCO(annotations)
catIds = coco_anns.loadCats(coco_anns.getCatIds())
imgIds = coco_anns.getImgIds([47])
annIds = coco_anns.getAnnIds(imgIds)
anns = coco_anns.loadAnns(annIds)
im = cv2.imread(image_path)
left_img = im[:,0:im.shape[1]//2,:]
right_img = im[:,im.shape[1]//2:im.shape[1],:]
print("Image shape {}".format(im.shape[1]))
x_length_half = im.shape[1]//2
color = (255,0,0)
for ann in anns:
    x = int(ann['bbox'][0])
    x_end = int(ann['bbox'][0]+ann['bbox'][2])
    y = int(ann['bbox'][1])
    y_end = int(ann['bbox'][1]+ann['bbox'][3])
    print("Start {} End {}".format(x,x_end))
    if x_end < x_length_half:
        #Fully left
        cv2.rectangle(left_img,(x,y),(x_end,y_end),color=color,thickness=2)
    else:
        if x>= x_length_half:
            cv2.rectangle(right_img,(x-x_length_half,y),(x_end-x_length_half,y_end),color=color,thickness=2)
        else:
            #split this to left and right annottaions
            #left split
            cv2.rectangle(left_img,(x,y),(x_length_half-1,y_end),color=color,thickness=2)
            cv2.rectangle(right_img,(0,y),(x_end-x_length_half,y_end),color=color,thickness=2)

cv2.imwrite('left_split.jpg',left_img)
cv2.imwrite('right_split.jpg',right_img)

    


