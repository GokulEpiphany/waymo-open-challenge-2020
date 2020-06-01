import os
import cv2
from pycocotools.coco import COCO
import argparse
import json

parser = argparse.ArgumentParser(description='Splitting the dataset')

parser.add_argument('data',metavar='DIR',help='folder containing train,valid,annotations subfolders')
parser.add_argument('--split',type=str,help='training or validation')
parser.add_argument('--output',type=str,help='folder to save')
args = parser.parse_args()
if args.split == 'train':
    img_path = os.path.join(args.data,'train')
    annotations_path = os.path.join(args.data,'annotations','train_annotations.json')
    img_save_dir = os.path.join(args.output,'train')
    annotation_file = os.path.join(args.output,'annotations','train_annotations.json')
else:
    img_path= os.path.join(args.data,'valid')
    annotations_path = os.path.join(args.data,'annotations','valid_small.json')
    img_save_dir = os.path.join(args.output,'valid')
    annotation_file = os.path.join(args.output,'annotations','valid_annotations.json')

output= args.output

info = { "year":2020,"version":"1.0","description":"Coco type dataset for waymo","contributor":"Gokul","url":"http://eloop.ai","date_created":"2020"}

WAYMO_CLASSES = ['TYPE_UNKNOWN', 'TYPE_VEHICLE', 'TYPE_PEDESTRIAN', 'TYPE_SIGN', 'TYPE_CYCLIST']
NEEDED_CLASSES = ['TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_CYCLIST']
licenses = [{"id":1,"name":"Non-commerical","url":"http://creativecommons.org/licenses/by-nc-sa/2.0"}]

images=[]
annotations=[]
saveId = 1
ignoreId = 1
annotationId = 1
waymo_type= "instances"
categories = [{"id":seqId+1,"name":waymo_class,"supercategory":waymo_class}for seqId,waymo_class in enumerate(NEEDED_CLASSES)]
cat2id = {cat["name"]:catId+1 for catId,cat in enumerate(categories)}

coco = COCO(annotations_path)
imgIds = coco.getImgIds()
for imgId in range(0,len(imgIds),2):
    chosenId = imgIds[imgId]
    print("Chosen {}".format(chosenId))
    im = cv2.imread(os.path.join(img_path,'{}.jpg'.format(chosenId)))
    print(im.shape)
    annIds = coco.getAnnIds(chosenId)
    anns = coco.loadAnns(annIds)
    x_length_half = im.shape[1]//2
    y_h = im.shape[0]
    check_height=False
    if y_h == 1280:
        check_height=True
        left_img = im[160:1120:,0:x_length_half,:]
        right_img = im[160:1120:,x_length_half:im.shape[1],:]
    else:
        left_img = im[:,0:x_length_half,:]
        right_img = im[:,x_length_half:im.shape[1],:]
    image = {"date_captured":"2020","file_name":str(saveId)+".jpg","id":saveId,"license":1,"url":"","height":left_img.shape[0],"width":left_img.shape[1]}
    images.append(image)
    
    image = {"date_captured":"2020","file_name":str(saveId+1)+".jpg","id":saveId+1,"license":1,"url":"","height":right_img.shape[0],"width":right_img.shape[1]}
    images.append(image)
    for ann in anns:
        x =(ann['bbox'][0])
        x_end =(ann['bbox'][0]+ann['bbox'][2])
        y =(ann['bbox'][1])
        y_end =(ann['bbox'][1]+ann['bbox'][3])
        if check_height:
            if y_end <=160:
                continue
            if y>=1120:
                continue
            if y<=160:
                y = 160
            if y_end>=1120:
                y_end=1120
            y = y - 160
            y_end = y_end - 160
        if x_end < x_length_half:
            width = x_end - x
            height = y_end - y
            bbox = [x,y,width,height]
            annotation = {"segmentation":[],"area":width*height,"iscrowd":0,"image_id":saveId,"bbox":bbox,"category_id":ann['category_id'],"id":annotationId}
            annotations.append(annotation)
            annotationId+=1
        else:
            if x>=x_length_half:
                x_start = x-x_length_half
                y_start = y
                x_end = x_end-x_length_half
                y_end = y_end
                width = x_end - x_start
                height = y_end - y_start
                bbox = [x_start,y_start,width,height]
                annotation = {"segmentation":[],"area":width*height,"iscrowd":0,"image_id":saveId+1,"bbox":bbox,"category_id":ann['category_id'],"id":annotationId}
                annotations.append(annotation)
                annotationId+=1
            else:
                x_left_start = x
                y_left_start = y
                x_left_end = x_length_half-1
                y_left_end=y_end
                width_left = x_left_end - x_left_start
                height_left = y_left_end - y_left_start
                bbox = [x_left_start,y_left_start, width_left,height_left]
                annotation = {"segmentation":[],"area":width_left*height_left,"iscrowd":0,"image_id":saveId,"bbox":bbox,"category_id":ann['category_id'],"id":annotationId}
                annotations.append(annotation)
                annotationId+=1
                x_right_start = 0
                y_right_start = y
                x_right_end = x_end-x_length_half
                y_right_end = y_end
                width_right = x_right_end - x_right_start
                height_right = y_right_end - y_right_start
                bbox  =[x_right_start,y_right_start,width_right,height_right]
                annotation = {"segmentation":[],"area":width_right*height_right,"iscrowd":0,"image_id":saveId+1,"bbox":bbox,"category_id":ann['category_id'],"id":annotationId}
                annotations.append(annotation)
                annotationId+=1

    cv2.imwrite(os.path.join(img_save_dir,'{}.jpg'.format(saveId)),left_img)
    cv2.imwrite(os.path.join(img_save_dir,'{}.jpg'.format(saveId+1)),right_img)
    saveId+=2

json_data = {"info":info,"images":images,"licenses":licenses,"type":waymo_type,"annotations":annotations,"categories":categories}

with open(annotation_file,"w") as jsonfile:
    json.dump(json_data,jsonfile,indent=4)
