import cv2
import os
data_path = '/root/waymo/data_2d_od/segment_15/train'
for i in range(1,500):
    image_path = os.path.join(data_path,'{}.jpg'.format(i))
    print(image_path)
    im = cv2.imread(image_path)
    im = cv2.resize(im,(768,768))
    cv2.imwrite(os.path.join('/root/waymo/data_2d_od/segment_15/train_resized','{}.jpg'.format(i)),im)

