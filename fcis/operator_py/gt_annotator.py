# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Haozhi Qi, Guodong Zhang
# --------------------------------------------------------

import cv2
import mxnet as mx
import numpy as np

from bbox.bbox_transform import bbox_overlaps, bbox_transform, remove_repetition
from bbox.bbox_regression import expand_bbox_regression_targets
from mask.mask_transform import intersect_box_mask
import cPickle


class GTAnnotatorOperator(mx.operator.CustomOp):
    def __init__(self):
        super(GTAnnotatorOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        gt_boxes = in_data[0].asnumpy()

        # include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.hstack((zeros, gt_boxes[:, :-1]))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        label = gt_boxes[:, 4]

        self.assign(out_data[0], req[0], all_rois)
        self.assign(out_data[1], req[1], label)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)


@mx.operator.register('gt_annotator')
class GTAnnotatorProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(GTAnnotatorProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['gt_boxes']

    def list_outputs(self):
        return ['rois_output', 'label_output']

    def infer_shape(self, in_shape):
        gt_boxes_shape = in_shape[0]

        rois = gt_boxes_shape[0]

        output_rois_shape = (rois, 5)
        label_shape = (rois, )

        return [gt_boxes_shape], \
               [output_rois_shape, label_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return GTAnnotatorOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
