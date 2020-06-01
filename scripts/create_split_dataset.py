import os
import cv2
from pycocotools.coco import COCO
import argparse

parser = argparse.ArgumentParser(description='Splitting the dataset')

parser.add_argument('data',metavar='DIR',help='folder containing train,valid,annotations subfolders')
parser.add_argument('--split',type=str,help='training or validation')
parser.add_argument('--output',type=str,help='folder to save')
args = parser.parse_args()
if args.split == 'train':
    img_path = os.path.join(args.data,'train')
    annotations_path = os.path.join(args.data,'annotations','train_annotations.json')
else:
    img_path= os.path.join(args.data,'valid')
    annotations_path = os.path.join(args.data,'annotations','valid_annotations.json')

output= args.output

coco = COCO(annotations_path)
imgIds = coco.getImgIds()
color=(255,0,0)
for imgId in imgIds:
    im = cv2.imread(os.path.join(img_path,'{}.jpg'.format(imgId)))
    annIds = coco.getAnnIds(imgId)
    anns = coco.loadAnns(annIds)
    x_length_half = im.shape[1]//2
    left_img = im[:,0:x_length_half,:]
    right_img = im[:,x_length_half:im.shape[1],:]
    for ann in anns:
        x = int(ann['bbox'][0])
        x_end = int(ann['bbox'][0]+ann['bbox'][2])
        y = int(ann['bbox'][1])
        y_end = int(ann['bbox'][1]+ann['bbox'][3])
        if x_end < x_length_half:
            cv2.rectangle(left_img,(x,y),(x_end,y_end),color=color,thickness=2)
        else:
            if x>=x_length_half:
                cv2.rectangle(right_img,(x-x_length_half,y),(x_end-x_length_half,y_end),color=color,thickness=2)
            else:
                cv2.rectangle(left_img,(x,y),(x_length_half-1,y_end),color=color,thickness=2)
                cv2.rectangle(right_img,(0,y),(x_end-x_length_half,y_end),color=color,thickness=2)
    cv2.imwrite(os.path.join(output,'{}_left.jpg'.format(imgId)),left_img)
    cv2.imwrite(os.path.join(output,'{}_right.jpg'.format(imgId)),right_img)

