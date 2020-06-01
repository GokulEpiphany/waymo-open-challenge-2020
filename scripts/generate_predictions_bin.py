from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
import argparse
from pycocotools.coco import COCO
import json

parser = argparse.ArgumentParser(description = 'Script for creating bins for submission')
parser.add_argument('--anno-file',type=str,help='Path to annotation file')
parser.add_argument('--results-file',type=str,help='Results json file containing predictions')
camera_name = [dataset_pb2.CameraName.FRONT,dataset_pb2.CameraName.FRONT_LEFT,dataset_pb2.CameraName.FRONT_RIGHT,dataset_pb2.CameraName.SIDE_LEFT,dataset_pb2.CameraName.SIDE_RIGHT]

label_name = [label_pb2.Label.TYPE_VEHICLE,label_pb2.Label.TYPE_PEDESTRIAN,label_pb2.Label.TYPE_CYCLIST]

def _create_pd(args):
    objects = metrics_pb2.Objects()
    coco = COCO(args.anno_file)
    with open(args.results_file) as data_file:
        data = data_file.read()
        data_content = json.loads(data)
    results_per_id = []
    current_predictions = []
    imgIds = coco.getImgIds()
    current = imgIds[0]
    for pred in data_content:
        if pred['image_id']==current:
            added=False
            bbox=pred['bbox']
            center_x = bbox[0]+ (bbox[2]/2)
            center_y = bbox[1]+ (bbox[3]/2)
            length = bbox[2]
            width =bbox[3]
            sub = [center_x,center_y,length,width,pred['score'],pred['category_id']]
            current_predictions.append(sub)
        else:
            results_per_id.append(current_predictions)
            current_predictions=[]
            current+=1
            added=True
    if not added:
        results_per_id.append(current_predictions)
    print(len(results_per_id))
    print(len(imgIds))
    print(imgIds[0])
    assert len(results_per_id) == len(imgIds)
    for idx in range(0,len(imgIds)):
        if (idx %10000==0):
            print(idx)
        imgId = imgIds[idx]
        predictions = results_per_id[imgId-1]
        o = metrics_pb2.Object()
        imgInfo = coco.loadImgs(imgId)
        o.context_name = imgInfo[0]['context']
        o.frame_timestamp_micros = imgInfo[0]['frame_timestamp_micros']
        o.camera_name = camera_name[imgInfo[0]['camera_name']-1]
        for pred in predictions:
            if pred[4]<0.05:
                continue
            box = label_pb2.Label.Box()
            box.center_x = pred[0]
            box.center_y = pred[1]
            box.center_z = 0
            box.length = pred[2]
            box.width = pred[3]
            box.height = 0
            o.object.box.CopyFrom(box)
            o.score = pred[4]
            o.object.id = 'id'
            o.object.type = label_name[pred[5]-1]
            objects.objects.append(o)
    f = open('valid.bin','wb')
    f.write(objects.SerializeToString())
    f.close()

if __name__=="__main__":
    args = parser.parse_args()
    _create_pd(args)
