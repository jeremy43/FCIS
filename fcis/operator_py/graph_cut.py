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
from graphcut.comp_max_flow import solve_masks_with_lists


class GraphCutOperator(mx.operator.CustomOp):
    def __init__(self):
        super(GraphCutOperator, self).__init__()

    def forward(self, is_train, req, in_data, out_data, aux):
        color_hist_list = in_data[0].asnumpy()
        texture_hist_list = in_data[1].asnumpy()
        flow_avg_list = in_data[2].asnumpy()
        neighbor_matrix = in_data[3].asnumpy()
        segment = in_data[4].asnumpy()
        network_term = in_data[5].asnumpy()     # it is seg_prob of shape # of ROI * 2 * 21 * 21
        gt_boxes = in_data[6].asnumpy() / 2
        im_tensor = in_data[7].asnumpy()
        epoch = in_data[8].asnumpy()

        im = np.zeros((im_tensor.shape[2], im_tensor.shape[3], 3))
        for i_rgb in range(3):
            im[:, :, 2 - i_rgb] = im_tensor[0, i_rgb, :, :] + 127

        assert np.sum(network_term) != 0, 'network term is all 0'


        num_segment = len(np.unique(segment))
        prob_term = np.ones((len(gt_boxes[0]), num_segment, 2)) / 2.0

        if epoch > 1:
            for idx, gt_box in enumerate(gt_boxes[0]):
                pos_prob_map = np.zeros(segment.shape)
                neg_prob_map = np.zeros(segment.shape)
                cod = np.zeros(4).astype(int)
                cod[0] = int(gt_box[0])
                cod[1] = int(gt_box[1])
                cod[2] = int(gt_box[2])
                cod[3] = int(gt_box[3])
                if segment[cod[1]:cod[3], cod[0]:cod[2]].size > 0:
                    pos_prob_map[cod[1]:cod[3], cod[0]:cod[2]] = cv2.resize(network_term[idx][1], segment[cod[1]:cod[3], cod[0]:cod[2]].T.shape)
                    neg_prob_map[cod[1]:cod[3], cod[0]:cod[2]] = cv2.resize(network_term[idx][0], segment[cod[1]:cod[3], cod[0]:cod[2]].T.shape)
                for i in range(num_segment):
                    mask = (segment == i)
                    sum_of_mask = np.sum(mask)
                    prob_term[idx][i][1] = np.sum(pos_prob_map[mask]) / sum_of_mask
                    prob_term[idx][i][0] = np.sum(neg_prob_map[mask]) / sum_of_mask

        gt_masks, gt_mask_weights, ratios = solve_masks_with_lists(color_hist_list, texture_hist_list,
                                                                   flow_avg_list, neighbor_matrix, segment,
                                                                   gt_boxes.astype(np.uint16), prob_term, im,
                                                                   use_flow=True, use_crf=False)

        gt_masks_resized = []
        # resize mask to double size
        for idx, mask in enumerate(gt_masks):
            if mask.shape != im[:,:,0].shape:
                mask = cv2.resize(mask, im[:,:,0].T.shape, interpolation=cv2.INTER_NEAREST)
            gt_masks_resized.append(mask)

        gt_masks_resized = mx.nd.array(gt_masks_resized)
        ratios = mx.nd.array(ratios)

        # train detections only
        if epoch == 0:
            gt_mask_weights = mx.nd.zeros((len(gt_mask_weights), ))
        else:
            gt_mask_weights = mx.nd.array(gt_mask_weights)

        self.assign(out_data[0], req[0], gt_masks_resized)
        self.assign(out_data[1], req[1], gt_mask_weights)
        self.assign(out_data[2], req[2], ratios)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        for i in range(len(in_grad)):
            self.assign(in_grad[i], req[i], 0)


@mx.operator.register('GraphCut')
class GraphCutProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(GraphCutProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['color_hist_list', 'texture_hist_list', 'flow_avg_list', 'neighbor_matrix', 'segment', 'network_term', 'gt_boxes', 'data', 'epoch']

    def list_outputs(self):
        return ['gt_masks', 'gt_mask_weights', 'ratios']

    def infer_shape(self, in_shape):
        output_shape = [in_shape[6][1]] + [shape for shape in in_shape[7][-2:]]
        weights_shape = [in_shape[6][1]]

        return in_shape, [output_shape, weights_shape, weights_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return GraphCutOperator()

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
