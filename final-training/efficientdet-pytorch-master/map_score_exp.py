import os
import numpy
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ann_file = '/root/waymo/split_dataset/annotations/valid_annotations.json'
coco = COCO(ann_file)
results = 'results.json'
coco_results = coco.loadRes(results)
coco_eval = COCOeval(coco,coco_results,'bbox')
coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2,200 ** 2], [200 ** 2, 500 ** 2], [500 ** 2, 1e5 ** 2]]
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()


