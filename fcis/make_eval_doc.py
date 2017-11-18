import os

def get_all_image_path(imageset_file, image_path):
    with open(imageset_file, 'r') as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
    img_basenames = [x[0] for x in lines]
    gt_img_ids = [int(x[1]) for x in lines]






if '__main__' == __name__:
    eval_path = './data/ILSVRC2015/ImageSets/VID_val_videos_eval.txt'
    image_path = './data/eval_pic'
    get_all_image_path(eval_path, image_path)