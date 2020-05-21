# waymo-open-challenge-2020

Rough steps to do to training 2d object detection using EfficientDet
1. Write a script to download training (32 segments) and validation ( 8 segments) tar files
2. Parse segments' tf record 
3. Create a COCO style dataset 
4. Convert effdet to use pretrained coco weights and adapt it for waymo dataset (Checkpointing etc)
5. Train

