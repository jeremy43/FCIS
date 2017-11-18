import sys
import os
import numpy as np
import pydensecrf.densecrf as dcrf
from graphcut.superpixel_cache import load_cache
from graphcut.comp_max_flow import solve_masks_with_lists

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

im_dir = '/home/yanrui/code/Share/msralab/ILSVRC2015/ResizedData_half'
cache_dir = '/home/yanrui/code/Share/FCIS_video/data/cache'

middle_name = 'VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00000000'
im_names = ['000010', '000030', '000050', '000070']

roidbs = [load_gt_roidb(config.dataset.dataset, image_set, config.dataset.root_path, config.dataset.dataset_path,
                        flip=config.TRAIN.FLIP)
          for image_set in image_sets]
roidb = merge_roidb(roidbs)
roidb = filter_roidb(roidb, config)

def get_neighbor_matrix(spixel_list):
    matrix_size = len(spixel_list)
    matrix = np.zeros((matrix_size, matrix_size))
    for spixel in spixel_list:
        for neighbor in spixel.neighbor:
            matrix[spixel.id][neighbor] = 1
            matrix[neighbor][spixel.id] = 1
    return matrix

segment = load_cache(os.path.join(cache_dir, middle_name, im_names) + '.pkl')
neighbor_matrix = get_neighbor_matrix(segment['spixel'])
color_hist_list = np.array([spixel.color_hist for spixel in segment['spixel']])
texture_hist_list = np.array([spixel.texture_hist for spixel in segment['spixel']])
flow_avg_list = np.array([spixel.flow_avg for spixel in segment['spixel']])
segments = segment['segment']

im = imread(os.path.join(im_dir, middle_name, im_names) + '.jpg')

solve_masks_with_lists(color_hist_list, texture_hist_list, flow_avg_list, neighbor_matrix, segments, gt_boxes, network_term)