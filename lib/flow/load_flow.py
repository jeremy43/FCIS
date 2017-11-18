# --------------------------------------------------------
#
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Shuhao Fu, Yuqing Zhu
# --------------------------------------------------------

import numpy as np
import cv2
import os


def get_flow(flow_file, is_norm=True):
    if os.path.exists(flow_file):
        with open(flow_file, 'rb') as f:
            magic = np.fromfile(f, np.float32, count=1)
            assert 202021.25 == magic, 'Magic number incorrect. Invalid .flo file'
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2 * w * h)
            # Reshape data into 3D array (columns, rows, bands)
            data2D = np.resize(data, (h[0], w[0], 2))

            if is_norm:
                norm = np.linalg.norm(data2D)
                data2D /= norm
            # plt.imshow(data2D[:,:,0])
            # plt.show()
        return data2D # list of 281 * 500 *2
    else:
        return 0
#
# def resize_flow(flow, target_size, max_size, stride=0):
#     """
#     only resize input image to target size and return scale
#     :param im: BGR image input by opencv
#     :param target_size: one dimensional size (the short side)
#     :param max_size: one dimensional max size (the long side)
#     :param stride: if given, pad the image to designated stride
#     :param interpolation: if given, using given interpolation method to resize image
#     :return:
#     """
#     flow_shape = flow.shape
#     flow_size_min = np.min(flow_shape[0:2])
#     flow_size_max = np.max(flow_shape[0:2])
#     flow_scale = float(target_size) / float(flow_size_min)
#     # prevent bigger axis from being more than max_size:
#     if np.round(flow_scale * flow_size_max) > max_size:
#         flow_scale = float(max_size) / float(flow_size_max)
#     flow = cv2.resize(flow, None, None, fx=flow_scale, fy=flow_scale)
#
#     if stride == 0:
#         return flow, flow_scale
#     else:
#         # pad to product of stride
#         flow_height = int(np.ceil(flow.shape[0] / float(stride)) * stride)
#         flow_width = int(np.ceil(flow.shape[1] / float(stride)) * stride)
#         flow_channel = flow.shape[2]
#         padded_flow = np.zeros((flow_height, flow_width, flow_channel))
#         padded_flow[:flow.shape[0], :flow.shape[1], :] = flow
#         return padded_flow, flow_scale

def draw_flow(flow):
    horz = cv2.normalize(flow[:,:, 0], None, 0, 255, cv2.NORM_MINMAX)
    vert = cv2.normalize(flow[:,:, 1], None, 0, 255, cv2.NORM_MINMAX)
    horz = horz.astype('uint8')
    vert = vert.astype('uint8')
    cv2.imshow('Horizontal Component', horz)
    cv2.imshow('Vertical Component', vert)