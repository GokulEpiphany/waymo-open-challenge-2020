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
    img_path= os.path.join(args.data,'test')
    annotations_path = os.path.join(args.data,'annotations','test_annotations.json')
    img_save_dir = os.path.join(args.output,'test')
    annotation_file = os.path.join(args.output,'annotations','test_annotations.json')

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
for imgId in range(0,len(imgIds)):
    chosenId = imgIds[imgId]
    print("Chosen {}".format(chosenId))
    im = cv2.imread(os.path.join(img_path,'{}.jpg'.format(chosenId)))
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
    image = {"date_captured":"2020","file_name":str(chosenId)+"_left.jpg","id":saveId,"license":1,"url":"","height":left_img.shape[0],"width":left_img.shape[1]}
    images.append(image)
    
    image = {"date_captured":"2020","file_name":str(chosenId)+"_right.jpg","id":saveId+1,"license":1,"url":"","height":right_img.shape[0],"width":right_img.shape[1]}
    images.append(image)

    cv2.imwrite(os.path.join(img_save_dir,'{}_left.jpg'.format(chosenId)),left_img)
    cv2.imwrite(os.path.join(img_save_dir,'{}_right.jpg'.format(chosenId)),right_img)
    saveId+=2

json_data = {"info":info,"images":images,"licenses":licenses,"type":waymo_type,"annotations":annotations,"categories":categories}

with open(annotation_file,"w") as jsonfile:
    json.dump(json_data,jsonfile,indent=4)
