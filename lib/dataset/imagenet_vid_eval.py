# --------------------------------------------------------
# Flow-Guided Feature Aggregation
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Xizhou Zhu
# --------------------------------------------------------

"""
given a imagenet vid imdb, compute mAP
"""

import numpy as np
import os
import cPickle
import cv2
from bbox.bbox_transform import bbox_overlaps



def parse_vid_rec(filename, classhash, img_ids, defaultIOUthr=0.5, pixelTolerance=10):
    """
    parse imagenet vid record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['label'] = classhash[obj.find('name').text]
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [float(bbox.find('xmin').text),
                            float(bbox.find('ymin').text),
                            float(bbox.find('xmax').text),
                            float(bbox.find('ymax').text)]
        gt_w = obj_dict['bbox'][2] - obj_dict['bbox'][0] + 1
        gt_h = obj_dict['bbox'][3] - obj_dict['bbox'][1] + 1
        thr = (gt_w*gt_h)/((gt_w+pixelTolerance)*(gt_h+pixelTolerance))
        obj_dict['thr'] = np.min([thr, defaultIOUthr])
        objects.append(obj_dict)
    # print 'reading annotations from file {}, there are {} objects'.format(filename, len(objects))
    return {'bbox' : np.array([x['bbox'] for x in objects]),
             'label': np.array([x['label'] for x in objects]),
             'thr'  : np.array([x['thr'] for x in objects]),
             'img_ids': img_ids}


def parse_vid_segm(filename):
    """
    parse imagenet vid record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    gt_masks = []
    mask_boxes = []
    size = tree.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    for obj_idx, obj in enumerate(tree.findall('polygon')):
        points = []
        for point in obj.findall('point'):
            points.append((float(point.find('X').text),
                            float(point.find('Y').text)))

        from PIL import ImageDraw, Image
        temp = Image.new(mode='P', size=(width, height))
        draw = ImageDraw.Draw(temp)
        draw.polygon((points), fill=obj_idx+1)
        mask = np.array(temp)
        
        coord = np.where(mask != 0)
        xmin = np.min(coord[1])
        xmax = np.max(coord[1])
        ymin = np.min(coord[0])
        ymax = np.max(coord[0])
        mask_box = [xmin, ymin, xmax, ymax]
        mask_boxes.append(mask_box)
        gt_masks.append(mask)

        assert mask[int(mask_box[1]):int(mask_box[3]), int(mask_box[0]):int(mask_box[2])].sum() != 0, 'mask sum is 0 encountered'

    return gt_masks, mask_boxes


def vid_ap(rec, prec):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def vid_eval(multifiles, detpath, segcache, seg_crf_file, graphcutcache, annopath, imageset_file, classname_map, annocache, ovthresh=0.5, mask_thresh=0.4, use_crf=True):
    """
    imagenet vid evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :return: rec, prec, ap
    """
    with open(imageset_file, 'r') as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
    img_basenames = [x[0] for x in lines]
    gt_img_ids = [int(x[1]) for x in lines]
    classhash = dict(zip(classname_map, range(0,len(classname_map))))

    # load annotations from cache
    if not os.path.isfile(annocache):
    # TODO: change it back
    # if True:
        recs = []
        maskpath = './data/Annotation/{0!s}.xml'
        for ind, image_filename in enumerate(img_basenames):
            rec = parse_vid_rec(annopath.format('VID/' + image_filename), classhash, gt_img_ids[ind])
            gt_mask, mask_box = parse_vid_segm(maskpath.format(image_filename))
            if len(gt_mask) != len(rec['bbox']):
                print 'numbers of gt_mask and bbox do not match in file id {}, len(gt_mask) = {} and len(bbox) = {}'.format(rec['img_ids'],  len(gt_mask), len(rec['bbox']))
            rec['mask'] = np.array(gt_mask)
            rec['mask_box'] = np.array(mask_box)

            # graph cut
            from graphcut.comp_max_flow import get_neighbor_matrix
            from graphcut.superpixel_cache import load_cache
            segment = load_cache(graphcutcache.format(image_filename))
            neighbor_matrix = get_neighbor_matrix(segment['spixel'])
            rec['color_hist'] = np.array([spixel.color_hist for spixel in segment['spixel']])
            rec['texture_hist'] = np.array([spixel.texture_hist for spixel in segment['spixel']])
            rec['flow_avg'] = np.array([spixel.flow_avg for spixel in segment['spixel']])
            rec['neighbor_matrix'] = neighbor_matrix
            rec['segment'] = segment['segment']

            recs.append(rec)
            if ind % 100 == 0:
                print 'reading annotations for {:d}/{:d}'.format(ind + 1, len(img_basenames))
        print 'saving annotations cache to {:s}'.format(annocache)
        with open(annocache, 'wb') as f:
            cPickle.dump(recs, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        with open(annocache, 'rb') as f:
            recs = cPickle.load(f)
    

    # extract objects in :param classname:
    npos = np.zeros(len(classname_map))
    for rec in recs:
        rec_labels = rec['label']
        for x in rec_labels:
            npos[x] += 1

    # read detections
    splitlines = []
    if (multifiles == False):
        with open(detpath, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
    else:
        for det in detpath:
            with open(det, 'r') as f:
                lines = f.readlines()
            splitlines += [x.strip().split(' ') for x in lines]

    # print 'det path is', detpath
    print 'reading masks from', segcache
    with open(segcache, 'rb') as f:
        segs = cPickle.load(f)

    if use_crf:
        with open(seg_crf_file, 'rb') as f:
            segs_crf = cPickle.load(f)


    img_ids = np.array([int(x[0]) for x in splitlines])
    obj_labels = np.array([int(x[1]) for x in splitlines])
    obj_confs = np.array([float(x[2]) for x in splitlines])
    obj_bboxes = np.array([[float(z) for z in x[3:]] for x in splitlines])
    obj_segs = np.array(segs)
    if use_crf:
        obj_crf_segs = np.array(segs_crf)
        assert len(obj_crf_segs) == len(img_ids), 'error with crf_segs\' shape'

    assert len(obj_segs) == len(img_ids), 'error with segs\' shape'

    # sort by confidence
    if obj_bboxes.shape[0] > 0:
        sorted_inds = np.argsort(img_ids)
        img_ids = img_ids[sorted_inds]
        obj_labels = obj_labels[sorted_inds]
        obj_confs = obj_confs[sorted_inds]
        obj_bboxes = obj_bboxes[sorted_inds, :]
        obj_segs = obj_segs[sorted_inds, :]
        if use_crf:
            obj_crf_segs = obj_crf_segs[sorted_inds, :]

    num_imgs = max(max(gt_img_ids),max(img_ids)) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    obj_segs_cell = [None] * num_imgs
    if use_crf:
        obj_crf_segs_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids)-1 or img_ids[i+1] != id:
            conf = obj_confs[start_i:i+1]
            label = obj_labels[start_i:i+1]
            bbox = obj_bboxes[start_i:i+1, :]
            seg = obj_segs[start_i:i+1, :]
            if use_crf:
                crf_seg = obj_crf_segs[start_i:i+1, :]
            sorted_inds = np.argsort(-conf)

            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            obj_segs_cell[id] = seg[sorted_inds, :]
            if use_crf:
                obj_crf_segs_cell[id] = crf_seg[sorted_inds, :]
            if i < len(img_ids)-1:
                id = img_ids[i+1]
                start_i = i+1


    # go down detections and mark true positives and false positives
    tp_cell = [None] * num_imgs
    fp_cell = [None] * num_imgs

    tp_seg_cell = [None] * num_imgs
    fp_seg_cell = [None] * num_imgs

    tp_gc_cell = [None] * num_imgs
    fp_gc_cell = [None] * num_imgs

    for rec in recs:
        id = rec['img_ids']
        gt_labels = rec['label']
        gt_bboxes = rec['bbox']
        gt_thr = rec['thr']
        mask = rec['mask']
        mask_box = rec['mask_box']
        overlaps = bbox_overlaps(mask_box.astype(np.float), gt_bboxes.astype(np.float))
        gt_assignment = overlaps.argmax(axis=1)
        gt_masks = np.zeros(mask.shape)
        for idx in range(len(gt_assignment)):
            assigned = gt_assignment[idx]
            gt_masks[assigned][int(gt_bboxes[assigned][1]):int(gt_bboxes[assigned][3]), int(gt_bboxes[assigned][0]):int(gt_bboxes[assigned][2])] = \
                mask[idx][int(gt_bboxes[assigned][1]):int(gt_bboxes[assigned][3]), int(gt_bboxes[assigned][0]):int(gt_bboxes[assigned][2])]
            assert gt_masks[assigned].sum() != 0, 'gt assignment error in file {}, index {}'.format(id, idx)

        # graph cut terms
        color_hist_list = rec['color_hist']
        texture_hist_list = rec['texture_hist']
        flow_avg_list = rec['flow_avg']
        neighbor_matrix = rec['neighbor_matrix']
        segments = rec['segment']

        num_gt_obj = len(gt_labels)
        gt_detected = np.zeros(num_gt_obj)

        assert len(np.unique(gt_assignment)) == num_gt_obj, 'segmentation assignment error in file {}'.format(id)

        labels = obj_labels_cell[id]
        bboxes = obj_bboxes_cell[id]
        masks = obj_segs_cell[id]
        if use_crf:
            crf_masks = obj_crf_segs_cell[id]

        num_obj = 0 if labels is None else len(labels)
        tp = np.zeros(num_obj)
        fp = np.zeros(num_obj)
        tp_seg = np.zeros(num_obj)
        fp_seg = np.zeros(num_obj)
        tp_gc = np.zeros(num_obj)
        fp_gc = np.zeros(num_obj)

        for j in range(0,num_obj):
            bb = bboxes[j, :]
            ovmax = -1
            kmax = -1
            for k in range(0,num_gt_obj):
                # print 'labels[j] = {}, gt_labels[k] = {}'.format(labels[j], gt_labels[k])
                if labels[j] != gt_labels[k]:
                    continue
                if gt_detected[k] > 0:
                    continue
                bbgt = gt_bboxes[k, :]
                bi=[np.max((bb[0],bbgt[0])), np.max((bb[1],bbgt[1])), np.min((bb[2],bbgt[2])), np.min((bb[3],bbgt[3]))]
                iw=bi[2]-bi[0]+1
                ih=bi[3]-bi[1]+1
                if iw>0 and ih>0:            
                    # compute overlap as area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                           (bbgt[2] - bbgt[0] + 1.) * \
                           (bbgt[3] - bbgt[1] + 1.) - iw*ih
                    ov=iw*ih/ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= gt_thr[k] and ov > ovmax:
                        ovmax=ov
                        kmax=k
            if kmax >= 0:
                tp[j] = 1
                gt_detected[kmax] = 1

                # segmentation evaluation
                mskgt = gt_masks[kmax, :, :] != 0
                assert np.sum(mskgt) > 0, 'ground truth mask does not exist'

                msk = masks[j, :, :]
                msk = cv2.resize(msk, (int(bb[2] - bb[0]), int(bb[3] - bb[1])))
                # TODO: pass config to mask_thresh
                bimsk = msk > mask_thresh
                m_msk = np.zeros(gt_masks[kmax, :, :].shape)
                m_msk[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] = bimsk
                # bimsk = m_msk[int(gt_bboxes[kmax][1]):int(gt_bboxes[kmax][3]), int(gt_bboxes[kmax][0]):int(gt_bboxes[kmax][2])].astype(np.bool)

                mskgt = mskgt.astype(np.bool)
                m_msk = m_msk.astype(np.bool)
                msk_overlap = np.sum(m_msk & mskgt).astype(float)
                msk_union = np.sum(m_msk | mskgt).astype(float)
                if msk_overlap / msk_union > gt_thr[kmax]:
                    tp_seg[j] = 1
                else:
                    fp_seg[j] = 1

                # graph cut evaluation
                # from graphcut.comp_max_flow import solve_masks_with_lists
                # num_segment = len(np.unique(segments))
                # prob_term = np.ones((1, num_segment, 2)) / 2.0
                # m_masks, _, _ = solve_masks_with_lists(color_hist_list, texture_hist_list, flow_avg_list,
                #            neighbor_matrix, segments, bb.reshape(1,1,4), prob_term)
                # m_mask = m_masks[0]
                # if m_mask.shape != mskgt.shape:
                #     m_mask = cv2.resize(m_mask, mskgt.T.shape, interpolation=cv2.INTER_NEAREST)
                # m_mask = m_mask.astype(np.bool)
                # msk_overlap = np.sum(m_mask & mskgt).astype(float)
                # msk_union = np.sum(m_mask | mskgt).astype(float)
                # if msk_union <= 0:
                #     print 'msk_union=0 encountered'
                #     print msk_union
                #     print mskgt.shape
                #     print mskgt
                # if msk_overlap / msk_union > gt_thr[kmax]:
                #     tp_gc[j] = 1
                # else:
                #     fp_gc[j] = 1

                # crf masks
                crf_msk = crf_masks[j, :, :]
                m_msk = np.zeros(gt_masks[kmax, :, :].shape)
                m_msk[int(bb[1]):int(bb[3]), int(bb[0]):int(bb[2])] = crf_msk
                m_msk = m_msk.astype(np.bool)
                msk_overlap = np.sum(m_msk & mskgt).astype(float)
                msk_union = np.sum(m_msk | mskgt).astype(float)
                if msk_overlap / msk_union > gt_thr[kmax]:
                    tp_gc[j] = 1
                else:
                    fp_gc[j] = 1
            else:
                fp[j] = 1
                fp_seg[j] = 1
                fp_gc[j] = 1

        tp_cell[id] = tp
        fp_cell[id] = fp
        tp_seg_cell[id] = tp_seg
        fp_seg_cell[id] = fp_seg
        tp_gc_cell[id] = tp_gc
        fp_gc_cell[id] = fp_gc

    tp_all = np.concatenate([x for x in np.array(tp_cell)[gt_img_ids] if x is not None])
    fp_all = np.concatenate([x for x in np.array(fp_cell)[gt_img_ids] if x is not None])
    tp_seg_all = np.concatenate([x for x in np.array(tp_seg_cell)[gt_img_ids] if x is not None])
    fp_seg_all = np.concatenate([x for x in np.array(fp_seg_cell)[gt_img_ids] if x is not None])
    tp_gc_all = np.concatenate([x for x in np.array(tp_gc_cell)[gt_img_ids] if x is not None])
    fp_gc_all = np.concatenate([x for x in np.array(fp_gc_cell)[gt_img_ids] if x is not None])
    obj_labels = np.concatenate([x for x in np.array(obj_labels_cell)[gt_img_ids] if x is not None])
    confs = np.concatenate([x for x in np.array(obj_confs_cell)[gt_img_ids] if x is not None])

    sorted_inds = np.argsort(-confs)
    tp_all = tp_all[sorted_inds]
    fp_all = fp_all[sorted_inds]
    tp_seg_all = tp_seg_all[sorted_inds]
    fp_seg_all = fp_seg_all[sorted_inds]
    tp_gc_all = tp_gc_all[sorted_inds]
    fp_gc_all = fp_gc_all[sorted_inds]
    obj_labels = obj_labels[sorted_inds]

    ap = np.zeros(len(classname_map))
    for c in range(1, len(classname_map)):
        # compute precision recall
        fp = np.cumsum(fp_all[obj_labels == c])
        tp = np.cumsum(tp_all[obj_labels == c])
        rec = tp / float(npos[c])
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print 'pre for c {} is {} and recall for it is {}'.format(c, prec, rec)
        ap[c] = vid_ap(rec, prec)
    ap = ap[1:]

    ap_seg = np.zeros(len(classname_map))
    for c in range(1, len(classname_map)):
        # compute precision recall
        tp = np.cumsum(tp_seg_all[obj_labels == c])
        fp = np.cumsum(fp_seg_all[obj_labels == c])
        rec = tp / float(npos[c])
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print 'pre for c {} is {} and recall for it is {}'.format(c, prec, rec)
        ap_seg[c] = vid_ap(rec, prec)
    ap_seg = ap_seg[1:]

    ap_gc = np.zeros(len(classname_map))
    for c in range(1, len(classname_map)):
        # compute precision recall
        tp = np.cumsum(tp_gc_all[obj_labels == c])
        fp = np.cumsum(fp_gc_all[obj_labels == c])
        rec = tp / float(npos[c])
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        # print 'pre for c {} is {} and recall for it is {}'.format(c, prec, rec)
        ap_gc[c] = vid_ap(rec, prec)
    ap_gc = ap_gc[1:]
    return ap, ap_seg, ap_gc
