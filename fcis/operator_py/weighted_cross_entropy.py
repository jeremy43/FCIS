# --------------------------------------------------------
# Fully Convolutional Instance-aware Semantic Segmentation
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Shuhao Fu
# --------------------------------------------------------

import mxnet as mx
import numpy as np

class WeightedCEOperator(mx.operator.CustomOp):
    def __init__(self, grad_scale, ignore_label, use_ignore):
        super(WeightedCEOperator, self).__init__()
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label
        self.grad_scale = float(grad_scale)

        self._softmax_out = None
        self._mask_weights = None
        self._label = None

    def forward(self, is_train, req, in_data, out_data, aux):
        seg_pred = in_data[0]
        mask_weights = in_data[1]
        label = in_data[2]

        # print seg_pred
        # print mask_weights

        softmax_out = mx.nd.softmax(seg_pred, axis=1)

        self._softmax_out = softmax_out.copy()
        self._mask_weights = mask_weights.copy()
        self._label = label.copy()

        label = label.asnumpy().astype('int32')
        label_zero = np.where(label != -1, 1-label, -1)
        label = np.concatenate((label_zero, label), axis=1)
        cls = softmax_out.asnumpy() + 1e-14
        cls_loss = np.where(label != -1, -label * np.log(cls), 0)

        assert label.shape == seg_pred.shape, 'shape error'

        cls_loss = mx.nd.array(cls_loss)
        label = mx.nd.array(label)

        self.assign(out_data[0], req[0], cls_loss)
        self.assign(out_data[1], req[1], softmax_out)
        self.assign(out_data[2], req[2], label)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        softmax_out = self._softmax_out.asnumpy()
        mask_weights = self._mask_weights.asnumpy()
        label = self._label.asnumpy().astype('int32')

        label_zero = np.where(label != -1, 1-label, -1)
        label = np.concatenate((label_zero, label), axis=1)

        grad = (softmax_out - label) * self.grad_scale / (softmax_out.shape[2] * softmax_out.shape[3]) * mask_weights

        grad = np.where(label != -1, grad, 0)
        # print 'mean', np.mean(grad)
        # print grad.std()

        grad = mx.nd.array(grad)

        self.assign(in_grad[0], req[0], grad)
        self.assign(in_grad[1], req[1], 0)
        self.assign(in_grad[2], req[2], 0)


@mx.operator.register('weighted_cross_entropy')
class WeightedCEProp(mx.operator.CustomOpProp):
    def __init__(self, grad_scale=1, ignore_label=-1, use_ignore=False):
        super(WeightedCEProp, self).__init__(need_top_grad=False)
        self.use_ignore = use_ignore
        self.ignore_label = ignore_label
        self.grad_scale = grad_scale

    def list_arguments(self):
        return ['seg_pred', 'mask_weights', 'label']

    def list_outputs(self):
        return ['ce_loss', 'softmax_out', 'label_out']

    def infer_shape(self, in_shape):
        output_shape = in_shape[0]
        return in_shape, [output_shape, output_shape, output_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return WeightedCEOperator(self.grad_scale, self.ignore_label, self.use_ignore)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
