from pycocotools.coco import COCO

coco = COCO('/root/waymo/final_dataset/annotations/valid_annotations.json')

imgIds = coco.getImgIds()
max=0
for imgId in imgIds:
    annIds = coco.getAnnIds(imgId)
    if len(annIds)>max:
        max = len(annIds)
print(max)
