from __future__ import print_function
import glob
import os
import argparse
from convert_tfrecord import extract_frame, WAYMO_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument('split', choices=['training', 'validation'])
parser.add_argument('--out-dir', default='./tmp')
parser.add_argument('--resize', default=0.5625, type=float)
args = parser.parse_args()

url_template = 'gs://waymo_open_dataset_v_1_2_0/{split}/{split}_%04d.tar'.format(split=args.split)
if args.split == 'training':
    num_segs = 2
elif args.split == 'validation':
    num_segs = 2

if not os.path.exists(args.out_dir):
    os.mkdir(args.out_dir)

clip_id = len(glob.glob('labels/*.txt'))
for seg_id in range(0, num_segs):
    flag = os.system('gsutil cp ' + url_template % seg_id + ' ' + args.out_dir)
    assert flag == 0, 'Failed to download segment %d. Make sure gsutil is installed'%seg_id
    os.system('cd %s; tar xf %s_%04d.tar'%(args.out_dir, args.split, seg_id))

    print("Segment %d done"%seg_id)
