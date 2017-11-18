# --------------------------------------------------------
#
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Shuhao Fu
# --------------------------------------------------------

"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import cv2
import numpy  as np
import time


class MaskWeightOperator(mx.operator.CustomOp):
    def __init__(self, grad_scale, use_ignore, ignore_label):
        super(MaskWeightOperator, self).__init__()
        self.grad_scale = grad_scale
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label

    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        label = in_data[1].asnumpy()
        weight = in_data[2]
        rois = in_data[3].asnumpy()
        gt_boxes = in_data[4]

        softmax_data = mx.nd.softmax(data=data, axis=1)

        # for gt_idx, gt_box in enumerate(gt_boxes[0].asnumpy()):
        #     for j in range(len(rois)):
        #         if np.array_equal(rois[j][1:], gt_box[:-1]):
        #             print 'my softmax output'
        #             print softmax_data[j].asnumpy()



        self.assign(out_data[0], req[0], in_data[0])
        self.assign(out_data[1], req[1], softmax_data)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('MaskWeight')
class MaskWeightProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1, use_ignore=True, ignore_label=-1):
        super(MaskWeightProp, self).__init__(need_top_grad=True)
        self.grad_scale = grad_scale
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label

    def list_arguments(self):
        return ['data', 'label', 'weight', 'rois', 'gt_boxes']

    def list_outputs(self):
        return ['data_out', 'seg_prob']

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]

        return in_shape, [output_shape, output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return MaskWeightOperator(self.grad_scale, self.use_ignore, self.ignore_label)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
