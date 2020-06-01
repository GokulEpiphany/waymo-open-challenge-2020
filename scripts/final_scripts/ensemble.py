import argparse
import json
from ensemble_boxes import *
from pycocotools.coco import COCO

parser = argparse.ArgumentParser(description='Apply ensemble')
parser.add_argument('--files',type=str,nargs='+')
parser.add_argument('--anno-file',type=str)

def main(args):
    coco = COCO(args.anno_file)
    imgIds = coco.getImgIds()
    file_list = args.files
    all_pred_results=[]
    all_score_results=[]
    all_label_results=[]
    for file in file_list:
        print("Parsing {}".format(file))
        results_per_id =[]
        scores_per_id=[]
        labels_per_id=[]
        current_predictions=[]
        current_scores=[]
        current_labels=[]
        with open(file) as data_file:
            data = data_file.read()
            data_content = json.loads(data)
        current = imgIds[0]
        imgInfo = coco.loadImgs(current)
        img_width = imgInfo[0]['width']
        img_height = imgInfo[0]['height']
        for pred in data_content:
            if pred['image_id']==current:
                added = False
                bbox=pred['bbox']
                x1 = bbox[0]
                y1 = bbox[1]
                x2 = bbox[0]+bbox[2]
                y2 = bbox[1]+bbox[3]
                x1 = x1/img_width
                y1 = y1/img_height
                x2 = x2/img_width
                y2 = y2/img_height
                sub = [x1,y1,x2,y2]
                current_predictions.append(sub)
                current_scores.append(pred['score'])
                current_labels.append(pred['category_id'])
            else:
                added=True
                results_per_id.append(current_predictions)
                scores_per_id.append(current_scores)
                labels_per_id.append(current_labels)
                current_predictions=[]
                current_scores=[]
                current_labels=[]
                current +=1
                imgInfo = coco.loadImgs(current)
                img_width = imgInfo[0]['width']
                img_height = imgInfo[0]['height']
        if not added:
            results_per_id.append(current_predictions)
            scores_per_id.append(current_scores)
            labels_per_id.append(current_labels)
        assert len(results_per_id) == len(imgIds)
        assert len(scores_per_id) == len(imgIds)
        assert len(labels_per_id) == len(imgIds)
        all_pred_results.append(results_per_id)
        all_score_results.append(scores_per_id)
        all_label_results.append(labels_per_id)
    boxes_list = []
    scores_list = []
    labels_list=[]
    weights = [1,1]
    final_results = []
    for idx in range(0,len(imgIds)):
        print("Image parsing {}".format(idx))
        given_id = imgIds[idx]
        boxes_list=[]
        scores_list=[]
        labels_list =[]
        imgInfo = coco.loadImgs(given_id)
        img_width = imgInfo[0]['width']
        img_height = imgInfo[0]['height']
        for temp in range(0,len(file_list)):
            boxes_list.append(all_pred_results[temp][idx])
            scores_list.append(all_score_results[temp][idx])
            labels_list.append(all_label_results[temp][idx])
        boxes,scores,labels =weighted_boxes_fusion(boxes_list,scores_list,labels_list,weights=weights,iou_thr=0.5,skip_box_thr=0.005)
        boxes = boxes.tolist()
        scores = scores.tolist()
        labels = labels.tolist()
        for box_id in range(0,len(boxes)):
            box = boxes[box_id]
            box[0] = box[0]*img_width
            box[1] = box[1]*img_height
            box[2] = (box[2]*img_width)-box[0]
            box[3] = (box[3]*img_height)-box[1]
            one_prediction = {"image_id":given_id,"bbox":box,"score":scores[box_id],"category_id":labels[box_id]}
            final_results.append(one_prediction)
    with open('check.json','w') as jsonfile:
        json.dump(final_results,jsonfile,indent=4)


if __name__=="__main__":
    args = parser.parse_args()
    main(args)
