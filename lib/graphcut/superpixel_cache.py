# --------------------------------------------------------
#
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Shuhao Fu, Yuqing Zhu
# --------------------------------------------------------

import cv2
import numpy as np


from multiprocessing.pool import ThreadPool as Pool
import cPickle as pickle
import os
import random
from scipy import ndimage
from skimage.segmentation import slic
from skimage.segmentation import find_boundaries
from skimage.util import img_as_float


DELTA_C = 0.6
DELTA_P = 0.0005
DELTA_T = 0.4
N_BG = 80

imageset_val_file = 'data/ILSVRC2015/ImageSets_Resized/VID_val_frames.txt'
imageset_train_file = 'data/ILSVRC2015/ImageSets_Resized/VID_train_15frames.txt'
ResizedData_path = '/home/v-shufu/code/msralab/ILSVRC2015/ResizedData_half/VID'
ResizedFlow_path = '/home/v-shufu/code/msralab/ILSVRC2015/ILSVRC2015'
temp_path = 'data/ILSVRC2015/ImageSets/temp.txt'
cache_path = 'data/cache'

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

def resize(im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :param stride: if given, pad the image to designated stride
    :param interpolation: if given, using given interpolation method to resize image
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

    if stride == 0:
        return im, im_scale
    else:
        # pad to product of stride
        im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
        im_width = int(np.ceil(im.shape[1] / float(stride)) * stride)
        im_channel = im.shape[2]
        padded_im = np.zeros((im_height, im_width, im_channel))
        padded_im[:im.shape[0], :im.shape[1], :] = im
        return padded_im, im_scale

def build_spixel(start, img_basenames, config=None):
    # spixel_cache=[]
    # segment_cache=[]

    import time
    start_time = time.time()
    for idx, image_filename in enumerate(img_basenames):
        segment_file = os.path.join(cache_path, 'VID', image_filename + '.pkl')
        if os.path.exists(segment_file):
            continue
        # if start == 0:
        #     print image_filename
        segment_res = dict()
        # load image
        image_time = time.time()
        current_path = os.path.join(ResizedData_path, image_filename + '.JPEG')
        if not os.path.exists(current_path):
            continue
        image = cv2.imread(current_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        # load flow
        from lib.flow.load_flow import get_flow
        list_index = image_filename.split('/')
        flow_index = 'forward_%d_%d' % (int(list_index[-1]), int(list_index[-1]) + 1)
        list_index = list_index[0:-1]
        list_index.append(flow_index)
        list_index = '/'.join(list_index)
        flow_file = os.path.join(ResizedFlow_path, 'Flownet2_half', 'VID', list_index + '.flo')
        if os.path.exists(flow_file):
            flow = get_flow(flow_file)
        else:
            flow = np.zeros((image.shape[0:2] + (2,)))
        assert flow.shape[0:2] == image.shape[0:2], 'flow shape and image shape are not consistent, flow shape: %s; image shape: %s' % (flow.shape, image.shape)

        # build super_pixel here and use pixel_p.neighbor to get the label of its neighbor
        segments = slic(img_as_float(image), n_segments=300, sigma=5)

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
            flow_seg = flow[segments == i]
            flow_avg = np.mean(flow_seg, axis=0)
            # create a new superpixel object
            new_spixel = super_p(i, np.array(color_hist), np.array(texture_hist), np.array(flow_avg))
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


        #print 'start=',start,' index=',idx
        segment_res['spixel'] = spixel_set
        segment_res['segment'] = segments
        # spixel_cache.append([start+idx,spixel_set])
        # segment_cache.append([start+idx,segments])

        if start == 0 and idx % 20 == 0:
            print 'finished %d images' % idx
            print 'file name: ', current_path
            print 'time consumed: ', time.time() - start_time

        with open(segment_file, 'wb') as f:
            pickle.dump(segment_res, f, protocol=pickle.HIGHEST_PROTOCOL)


    # return spixel_cache, segment_cache

def get_image(image_path, config=None):
    im = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    if config:
        scale_ind = random.randrange(len(config.SCALES))
        target_size = config.SCALES[scale_ind][0]
        max_size = config.SCALES[scale_ind][1]
        stride = config.network.IMAGE_STRIDE
    else:
        target_size = 600
        max_size = 1000
        stride = 0
    im, im_scale = resize(im, target_size, max_size, stride=stride)
    return im


def clear_images(is_train=True):
    temp_path = 'data/ILSVRC2015/ImageSets/temp.txt'
    imageset_file = imageset_train_file if is_train else imageset_val_file
    print 'loading imageset from', imageset_file
    with open(imageset_file, 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
        print 'number of images', len(lines)
    if len(lines[0]) == 2:
        image_set_index = [x[0] for x in lines]
        frame_id = [x[1] for x in lines]
        with open(temp_path, 'wb') as f:
            for idx, image_filename in enumerate(image_set_index):
                current_path = os.path.join(ResizedData_path, image_filename + '.JPEG')
                if os.path.exists(current_path):
                    f.write(image_filename + ' ' + frame_id[idx] + '\n')
    else:
        image_set_index = ['%s/%06d' % (x[0], int(x[2])) for x in lines]
        print len(image_set_index)
        pattern = [x[0]+'/%06d' for x in lines]
        frame_id = [int(x[1]) for x in lines]
        frame_seg_id = [int(x[2]) for x in lines]
        frame_seg_len = [int(x[3]) for x in lines]
        with open(temp_path, 'wb') as f:
            for idx, image_filename in enumerate(image_set_index):
                current_path = os.path.join(ResizedData_path, image_filename + '.JPEG')
                if os.path.exists(current_path):
                    f.write(' '.join(lines[idx]) + '\n')





def make_cache(is_train=True):
    # imageset_file = imageset_train_file if is_train else imageset_val_file
    imageset_file = temp_path
    print 'loading imageset from', imageset_file
    with open(imageset_file, 'r') as f:
        lines = [x.strip().split(' ') for x in f.readlines()]
    if len(lines[0]) == 2:
        image_set_index = [x[0] for x in lines]
        frame_id = [int(x[1]) for x in lines]
        for x in lines:
            if not os.path.exists(os.path.join(cache_path, 'VID', x[0])):
                os.makedirs(os.path.join(cache_path, 'VID', x[0]))
    else:
        image_set_index = ['%s/%06d' % (x[0], int(x[2])) for x in lines]
        pattern = [x[0]+'/%06d' for x in lines]
        frame_id = [int(x[1]) for x in lines]
        frame_seg_id = [int(x[2]) for x in lines]
        frame_seg_len = [int(x[3]) for x in lines]
        for x in lines:
            if not os.path.exists(os.path.join(cache_path, 'VID', x[0])):
                os.makedirs(os.path.join(cache_path, 'VID', x[0]))
    img_basenames = image_set_index

    print 'number of images', len(img_basenames)

    # from multiprocessing.pool import ThreadPool as Pool
    from multiprocessing.pool import Pool as Pool
    # number of thread
    thread_num=32
    img_per_thread=len(img_basenames)/thread_num
    # partion of imageset , partion of flow-set is still needed
    per_imgset=[]
    start=0
    for i in range(thread_num-1):
        per_imgset.append(img_basenames[start:min(len(img_basenames)-1,start+img_per_thread)])
        start+=img_per_thread
    per_imgset.append(img_basenames[start:])
    print [len(per_imgset[i]) for i in range(len(per_imgset))]
    assert np.sum([len(per_imgset[i]) for i in range(len(per_imgset))]) == len(img_basenames), 'splitting error'
    print 'splitted data and start building superpixels'

    pool = Pool(processes=thread_num)
    multiple_results = [pool.apply_async(build_spixel, args=(i*img_per_thread,per_imgset[i])) for i
                        in range(thread_num)]

    pool.close()
    pool.join()
    # cache_spixel=[]
    # cache_segment=[]
    # for x in multiple_results:
    #     index=list(x.get())
    #     cache_spixel+=index[:][0]
    #     cache_segment+=index[:][1]
    #
    # #sort according to the image_idx
    # cache_spixel.sort(key=lambda x: x[0])
    # cache_segment.sort(key=lambda x: x[0])
    # #form of cache_spixel
    # '''
    #     the list of [image_idx,segment_set]
    #     segment_set is a set of instance spixel
    #     when we refer to the j^{th} spixel of  i^{th} image, we can refer to cache_spixel[i][1][j]
    #     cache_spixel[i][0] should be i
    # '''
    #
    # spixel_file = 'data/cache/spixel_cache_train.pkl' if is_train else 'data/cache/spixel_cache_val.pkl'
    # segment_file = 'data/cache/segment_cache_train.pkl' if is_train else 'data/cache/segment_cache_val.pkl'
    # with open(spixel_file, 'wb') as f:
    #     pickle.dump(cache_spixel, f, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open(segment_file, 'wb') as f:
    #     pickle.dump(cache_segment, f, protocol = pickle.HIGHEST_PROTOCOL)

def load_cache(cache_file):
    assert os.path.exists(cache_file), '{} does not exist'.format(cache_file)
    with open(cache_file, 'rb') as fid:
        segment = pickle.load(fid)
    return segment



if '__main__' == __name__:
    clear_images(is_train=True)
    # make_cache(is_train=False)
