# --------------------------------------------------------
#
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Shuhao Fu, Yuqing Zhu
# --------------------------------------------------------

import cv2
import numpy as np
import maxflow
import math
import sys
import os
from scipy import ndimage
from skimage.segmentation import slic
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float
from matplotlib import pyplot as plt
import time


DELTA_C = 1.2       #0.6
DELTA_P = 0.03        #0.0005
DELTA_T = 1.0       #0.4
N_BG = 3.0           #80
DELTA_NET = 2.0

SIG = 0.2
C = 0.6


class super_p():
    def __init__(self, id, color_hist=None, texture_hist=None, flow_avg=None):
        self.id = id
        self.neighbor = []
        self.color_hist = color_hist
        self.texture_hist = texture_hist
        self.flow_avg = flow_avg

    def add_neighbor(self,neighbor):
        self.neighbor = neighbor

    def __getstate__(self):
        odict = self.__dict__.copy()
        return odict

    def __setstate__(self, dict):
        self.__dict__.update(dict)  # update attributes

# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def get_color_histgram(image, mask=None, num_bins=25, is_norm=True):
    c_hist = []
    for ch in xrange(3):
        hist = cv2.calcHist([image], [ch], mask, [num_bins], [0.0,255.0])
        c_hist = c_hist + [element[0] for element in hist]

    if is_norm:
        norm = np.linalg.norm(c_hist)
        c_hist /= np.maximum(norm, np.finfo(np.float64).eps)
    return c_hist


def get_texture_histgram(image, mask=None, num_bins=10, is_norm=True):
    t_hist = []
    # image = image.astype(float)
    grad = [ndimage.sobel(image, 0), ndimage.sobel(image, 1)]
    for index in xrange(len(grad)):
        for ch in range(3):
            hist = cv2.calcHist([grad[index]], [ch], mask, [num_bins], [0.0,255.0])
            t_hist = t_hist + [element[0] for element in hist]

    if is_norm:
        norm = np.linalg.norm(t_hist)
        t_hist /= np.maximum(norm, np.finfo(np.float64).eps)
    return  t_hist

def get_neighbor_matrix(spixel_list):
                matrix_size = len(spixel_list)
                matrix = np.zeros((matrix_size, matrix_size))
                for spixel in spixel_list:
                    for neighbor in spixel.neighbor:
                        matrix[spixel.id][neighbor] = 1
                        matrix[neighbor][spixel.id] = 1
                return matrix

def build_spixel(image, flow):

    # build super_pixel here and use pixel_p.neighbor to get the label of its neighbor
    segments = slic(img_as_float(image), n_segments=300, sigma=5)

    ############################################################
    # converted_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # prior = 3
    # num_levels = 8
    # num_histogram_bins = 5
    # num_iterations = 4
    # num_superpixels = 300
    #
    # height, width, channels = converted_img.shape
    # seeds = cv2.ximgproc.createSuperpixelSEEDS(width, height, channels,
    #                                            num_superpixels, num_levels, prior, num_histogram_bins)
    #
    # color_img = np.zeros((height, width, 3), np.uint8)
    # color_img[:] = (0, 0, 255)
    #
    # seeds.iterate(converted_img, num_iterations)
    #
    # # retrieve the segmentation result
    # segments = seeds.getLabels()
    #################################################################

    # show the output of SLIC
    # spixel_image = mark_boundaries(img_as_float(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), segments)
    # plt.imshow(spixel_image)
    # plt.show()

    spixel_set = []
    bound_pixel = find_boundaries(segments,connectivity=1,mode='thick')
    num_segment = len(np.unique(segments))
    for i in range(num_segment):
        # get mask of this superpixel
        mask = (segments == i).astype(np.uint8)

        # generate color histgram and texture histgram for this superpixel
        color_hist = get_color_histgram(image, mask)
        texture_hist = get_texture_histgram(image, mask)

        # get average optical flow
        if flow != None:
            flow_seg = flow[segments == i]
            flow_avg = np.mean(flow_seg, axis=0)
            # create a new superpixel object
            new_spixel = super_p(i, np.array(color_hist), np.array(texture_hist), np.array(flow_avg))
        else:
            new_spixel = super_p(i, np.array(color_hist), np.array(texture_hist))
        spixel_set.append(new_spixel)


    neibor = [[]for i in range(num_segment)]
    boundindex = np.where(bound_pixel == 1)
    for index, x in enumerate(boundindex[0]):
        y = boundindex[1][index]
        cur_label = segments[x, y]
        if y < image.shape[1]-1 and segments[x, y+1] != cur_label:
            neibor[cur_label].append(segments[x, y+1])
            neibor[segments[x, y+1]].append(cur_label)
        if x < image.shape[0]-1 and segments[x+1, y] != cur_label:
            neibor[cur_label].append(segments[x+1, y])
            neibor[segments[x+1, y]].append(cur_label)

    for i, neighbor in enumerate(neibor):
        spixel_set[i].add_neighbor(np.unique(neighbor))

    return spixel_set, segments


def solve_masks_with_lists(color_hist_list, texture_hist_list, flow_avg_list,
                           neighbor_matrix, segments, gt_boxes, network_term,
                           image=None, use_flow=True, use_crf=False):
    # if not os.path.exists('data/test_parameter'):
    #        os.mkdir('data/test_parameter')
    # image_name = 'image{}{}{}'.format(int(image[100][100][0]),int(image[100][100][1]),int(image[100][100][2]))
    # save_data_path = os.path.join('data/test_parameter', image_name)
    # while os.path.exists(save_data_path):
    #     save_data_path += '_new'
    # if not os.path.exists(save_data_path):
    #    os.mkdir(save_data_path)
    num_nodes = len(color_hist_list)

    if use_flow:
        norm = np.linalg.norm(flow_avg_list, axis=0)
        flow_avg_list /= np.maximum(norm, np.finfo(np.float64).eps)

    mask_set = []
    weight_set = []
    ratio_set = []
    for box_idx, box in enumerate(gt_boxes[0]):
        # cv2.imwrite(os.path.join(save_data_path, image_name + '.jpg'), image)
        # for N_BG in [2.0, 5.0, 10.0]:
        #     for DELTA_C in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.5, 2.0, 3.0]:
        #         for DELTA_P in [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.5]:
        #             for DELTA_T in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.0, 1.2, 1.5]:

        # set edge weights
        g = maxflow.Graph[float](2, 6 * num_nodes)
        nodes = g.add_nodes(num_nodes)
        box_area = 0
        for index in range(num_nodes):
            # set unary terms
            edge_weight_t = 1 / float(N_BG) - math.log(np.maximum(network_term[box_idx][index][0], np.finfo(np.float64).eps)) / float(DELTA_NET)
            mask = (segments == index)
            sum_mask = np.sum(mask)
            box = np.array(box, dtype=int)
            mask[box[1]:box[3], box[0]:box[2]] = 0

            # larger than 1/10 part of the superpixel is out of the box
            if np.sum(mask) / sum_mask > 0.1:
                edge_weight_s = sys.maxsize
            else:
                box_area += sum_mask
                edge_weight_s = - math.log(np.maximum(network_term[box_idx][index][1], np.finfo(np.float64).eps)) / float(DELTA_NET)
            g.add_tedge(index, edge_weight_s, edge_weight_t)

            # set pairwise terms
            for x in range(len(neighbor_matrix[index])):
                if neighbor_matrix[index][x] == 1:
                    # subtraction of color histgrams
                    color_hist_sub = color_hist_list[x] - color_hist_list[index]
                    edge_weight_color = -np.inner(color_hist_sub, color_hist_sub) / float(DELTA_C ** 2)

                    # subtraction of texture histgrams
                    texture_hist_sub = texture_hist_list[x] - texture_hist_list[index]
                    edge_weight_text = -np.inner(texture_hist_sub, texture_hist_sub) / float(DELTA_T ** 2)

                    if use_flow:
                        flow_sub = flow_avg_list[x] - flow_avg_list[index]
                        edge_weight_flow = -np.inner(flow_sub, flow_sub) / float(DELTA_P ** 2)
                    else:
                        edge_weight_flow = 0

                    edge_weight = math.exp(edge_weight_color + edge_weight_text + edge_weight_flow)
                    g.add_edge(nodes[index], nodes[x], edge_weight, edge_weight)

        g.maxflow()
        label = []
        for i in xrange(num_nodes):
            label.append(g.get_segment(nodes[i]))
        label = np.array(label)

        mask = np.zeros(segments.shape, dtype=np.uint8)
        for seg_id in range(len(label)):
            if label[seg_id]:
                mask |= (segments == seg_id)

        final_mask = np.zeros(mask.shape)
        final_mask[box[1]:box[3], box[0]:box[2]] = mask[box[1]:box[3], box[0]:box[2]]
        mask = final_mask

        box_area += 1e-12
        ratio = np.sum(mask) / float(box_area)
        weight = comp_weight(ratio, SIG, C)

        # if mask.shape != image[:,:,0].shape:
        #         mask = cv2.resize(mask, image[:,:,0].T.shape)
        # masked_im = image * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        # cv2.imwrite(os.path.join(save_data_path, image_name + '_bg_{}_c_{}_t_{}_p_{}.jpg'.format(N_BG, DELTA_C, DELTA_T, DELTA_P)), masked_im)

        if use_crf:
            #####################################
            # testing CRF
            #####################################
            import pydensecrf.densecrf as dcrf
            from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

            mask = cv2.resize(mask, image[:,:,0].T.shape, interpolation=cv2.INTER_NEAREST)
            masked_im = image * np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            mask += 1

            box = box * 2
            cod = np.zeros(4).astype(int)
            cod[0] = int(box[0])
            cod[1] = int(box[1])
            cod[2] = int(box[2])
            cod[3] = int(box[3])
            mask_area = mask[cod[1]:cod[3], cod[0]:cod[2]]
            image_area = image[cod[1]:cod[3], cod[0]:cod[2], :]

            colors, labels = np.unique(mask_area, return_inverse=True)
            HAS_UNK = 0 in colors
            if HAS_UNK:
                print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
                print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")

            n_labels = 2

            # Example using the DenseCRF class and the util functions
            d = dcrf.DenseCRF(image_area.shape[1] * image_area.shape[0], n_labels)

            # get unary potentials (neg log probability)
            U = unary_from_labels(labels, n_labels, gt_prob=0.6, zero_unsure=HAS_UNK)
            d.setUnaryEnergy(U)

            # This creates the color-independent features and then add them to the CRF
            feats = create_pairwise_gaussian(sdims=(1, 1), shape=image_area.shape[:2])
            d.addPairwiseEnergy(feats, compat=1,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # This creates the color-dependent features and then add them to the CRF
            feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                              img=image_area, chdim=2)
            d.addPairwiseEnergy(feats, compat=3,
                                kernel=dcrf.FULL_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # Run five inference steps.
            Q = d.inference(5)

            # Find out the most probable class for each pixel.
            MAP = np.argmax(Q, axis=0)

            # Convert the MAP (labels) back to the corresponding colors and save the image.
            # Note that there is no "unknown" here anymore, no matter what we had at first.
            m_msk = MAP.reshape(image_area[:,:,0].shape)
            msk = np.zeros(image[:,:,0].shape)
            msk[cod[1]:cod[3], cod[0]:cod[2]] = m_msk
            mask = msk
            ratio = np.sum(mask) / float((cod[3] - cod[1]) * (cod[2] - cod[0]))
            weight = comp_weight(ratio, SIG, C)
			
            # debug
            #mskd_im = image * np.repeat(msk[:, :, np.newaxis], 3, axis=2)
            #index = 0
            #while os.path.exists(os.path.join(save_data_path, str(index) + '.jpg')):
            #    index += 1
            #cv2.imwrite(os.path.join(save_data_path, str(index) + '.jpg'), image)
            #cv2.imwrite(os.path.join(save_data_path, str(index) + '_masked.jpg'), masked_im)
            #cv2.imwrite(os.path.join(save_data_path, str(index) + '_out.jpg'), mskd_im)
			
            ############################
            # FINISHED
            ############################

        mask_set.append(mask)
        weight_set.append(weight)
        ratio_set.append(ratio)
    return mask_set, weight_set, ratio_set


def comp_weight(x, sig, c):
    return np.exp(- (x - c)**2 / (2 * sig**2))
