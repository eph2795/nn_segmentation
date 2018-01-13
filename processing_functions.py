
# coding: utf-8

# In[11]:

import numpy as np
from skimage.morphology import closing, square
from skimage.measure import label

import theano.tensor as T
from lasagne.layers import InputLayer
from lasagne.layers import Conv2DLayer as ConvLayer

from collections import OrderedDict
import matplotlib.pyplot as plt


# In[4]:

def predict_image(X, background, predictor, patch_size=128, n_input_chanels=1):    
    hs = [i * patch_size for i in range(X.shape[1] // patch_size)]
    if X.shape[1] % patch_size != 0:
        hs.append(X.shape[1] - patch_size)
    
    ws = [i * patch_size for i in range(X.shape[2] // patch_size)]
    if X.shape[2] % patch_size != 0:
        ws.append(X.shape[2] - patch_size)

    out_X = np.zeros((X.shape[1], X.shape[2]), dtype=np.int8)

    for h in hs:
        for w in ws:
            if np.sum(background[h:h + patch_size, w:w + patch_size]) == 0:
                out_X[h:h + patch_size, w:w + patch_size] = np.zeros((patch_size, patch_size))
                continue
            
            preds = predictor(
                X[:, h:h + patch_size, w:w + patch_size].reshape(1, n_input_channels, patch_size, patch_size))
            
            preds -= np.min(preds)
            if np.max(preds) != 0:
                preds /= np.max(preds)
            
            out_X[h:h + patch_size, w:w + patch_size] = preds[:, 0].reshape(patch_size, patch_size)
           
    
    return out_X * background


# In[61]:

def build_background_detector(BATCH_SIZE=None, input_dim=(1000, 1000), 
                              filter_size=11, threshold=0.1):
    background_detector = OrderedDict()
    background_detector['input'] = InputLayer((BATCH_SIZE, 1, input_dim[0], input_dim[1]))

    background_detector['background_detector'] = ConvLayer(
        background_detector['input'], num_filters=1, filter_size=filter_size, 
        nonlinearity=lambda x: T.where(T.le(x, threshold), 0, 1),
        pad='same', W=1.0 / filter_size ** 2 * np.ones((1, 1, filter_size, filter_size)), b=None)
    
    return background_detector


# In[363]:

def process_image(image, net, get_rough_background, get_accurate_background, patch_size=128, n_input_chanels=1,
                  binarization_threshold=0.3, closing_filter_size=8, minimal_crack_size=100):
    image_h = image.shape[0]
    image_w = image.shape[1]
    
    plt.figure(figsize=(16, 24))
    plt.subplot(3, 2, 1)
    plt.imshow(image, cmap='gray')
    
    rough_background = get_rough_background(image[np.newaxis, np.newaxis, :, :] + 0.5)
    accurate_background = get_accurate_background(rough_background)   
    plt.subplot(3, 2, 2)
    plt.imshow(accurate_background[0, 0], cmap='gray')
    
    prediction = predict_image(image[np.newaxis, :, :], accurate_background[0, 0], 
                               net, patch_size, n_input_chanels)
    plt.subplot(3, 2, 3)
    plt.imshow(prediction, cmap='gray')

    binarized = np.where(prediction > binarization_threshold, 1, 0)
    plt.subplot(3, 2, 4)
    plt.imshow(binarized, cmap='gray')
    
    filtered = closing(binarized, square(closing_filter_size))
    plt.subplot(3, 2, 5)
    plt.imshow(filtered, cmap='gray')
    
    blobs_labels = label(filtered, background=0)
    labels, counts = np.unique(blobs_labels, return_counts=True)
    final = np.where(np.in1d(blobs_labels, 
                             np.where(counts <= minimal_crack_size)[0]).reshape((image_h, image_w)), 0, filtered)
    plt.subplot(3, 2, 6)
    plt.imshow(np.where(np.logical_and(accurate_background[0, 0] == 1, final == 0), 0.4, final), cmap='gray')
    
    plt.show()
    
    bins = np.arange(0, 180, 5)
    area, length, weights = get_statistics(final, bins.size)
    plt.hist(bins, bins=bins.size, weights=weights)
    plt.show()
    print 'area: ', area
    print 'length: ', length
    return final, (bins, weights)


# In[ ]:

def process_image_old(image, net, get_rough_background, get_accurate_background, patch_size=128, n_input_chanels=1,
                  binarization_threshold=0.3, closing_filter_size=8, minimal_crack_size=100):
    image_h = image.shape[0]
    image_w = image.shape[1]
    
    plt.figure(figsize=(16, 24))
    plt.subplot(3, 2, 1)
    plt.imshow(image, cmap='gray')
    
    rough_background = get_rough_background(image[np.newaxis, np.newaxis, :, :] + 0.5)
    accurate_background = get_accurate_background(rough_background)   
    plt.subplot(3, 2, 2)
    plt.imshow(accurate_background[0, 0], cmap='gray')
    
    prediction = predict_image(image[np.newaxis, :, :], accurate_background[0, 0], 
                               net, patch_size, n_input_chanels)
    plt.subplot(3, 2, 3)
    plt.imshow(prediction, cmap='gray')

    binarized = np.where(prediction > binarization_threshold, 1, 0)
    plt.subplot(3, 2, 4)
    plt.imshow(binarized, cmap='gray')
    
    filtered = closing(binarized, square(closing_filter_size))
    plt.subplot(3, 2, 5)
    plt.imshow(filtered, cmap='gray')
    
    blobs_labels = label(filtered, background=0)
    labels, counts = np.unique(blobs_labels, return_counts=True)
    final = np.where(np.in1d(blobs_labels, 
                             np.where(counts <= minimal_crack_size)[0]).reshape((image_h, image_w)), 0, filtered)
    plt.subplot(3, 2, 6)
    plt.imshow(np.where(np.logical_and(accurate_background[0, 0] == 1, final == 0), 0.4, final), cmap='gray')
    
    plt.show()
    
    bins = np.arange(0, 180, 5)
    area, length, weights = get_statistics_old(final, bins.size)
    plt.hist(bins, bins=bins.size, weights=weights)
    plt.show()
    print 'area: ', area
    print 'length: ', length
    return final, (bins, weights)


# In[362]:

def get_statistics_old(image, n_bins=38):
    blobs_labels = label(image, background=0)
    labels, counts = np.unique(blobs_labels, return_counts=True)
    
    n = labels.size
    image_h = image.shape[0]
    image_w = image.shape[1]
    
    x = [[] for i in range(n)]
    y = [[] for i in range(n)]
    for i in range(image_h):
        for j in range(image_w):
            label = blobs_labels[i, j]
            x[label].append(i)
            y[label].append(j)

    for i in range(n):
        x[i] = np.array(x[i])
        y[i] = np.array(y[i])
        
    weight = np.zeros(n_bins, np.float64)
    area = 0.0
    length = 0.0
    for i in range(1, n):
        m = x[i].size
        cur_weights = np.zeros(n_bins, np.float64)
   
        area += m
        cur_length = 0.0
        for j in range(m):
            if blobs_labels[x[i][j] - 1:x[i][j] + 2, y[i][j] - 1:y[i][j] + 2].sum() != 9 * i:
                cur_length += 1
        cur_length *= 0.5
        length += cur_length
            
    
        for j in range(m):
            for k in range(m):
                angle = np.arctan2(y[i][j] - y[i][k], x[i][j] - x[i][k])
                if angle < 0:
                    angle += np.pi
                angle = angle / np.pi * 180 - 1e-3
            
                f = True 
                x0 = float(x[i][k])
                y0 = float(y[i][k])
                if (angle < 45) or (angle > 135):
                    dy = float((y[i][j] - y[i][k]) / (x[i][j] - x[i][k]))
                    dx = float(1)
                    for t in range(x[i][j] - x[i][k]):
                        if blobs_labels[int(x0), int(y0)] != i:
                            f = False
                            break
                        x0 += dx
                        y0 += dy
                else:
                    dy = float(1)
                    dx = float((x[i][j] - x[i][k]) / (y[i][j] - y[i][k]))
                    for t in range(y[i][j] - y[i][k]):
                        if blobs_labels[int(x0), int(y0)] != i:
                            f = False
                            break
                        x0 += dx
                        y0 += dy

                if f:
                    cur_weights[int(angle // 5)] += np.sqrt((y[i][j] - y[i][k]) ** 2 + (x[i][j] - x[i][k]) ** 2)
        cur_weights = cur_length * cur_weights / np.sum(cur_weights)
        weight += cur_weights
        
    return area, length, weight


# In[ ]:

def distance(one, another):
    return np.sqrt((one[0] - another[0]) ** 2 + (one[1] - another[1]) ** 2)
    
def angle(one, another):
    angle = np.arctan2(another[1] - one[1], another[0] - one[0])
    if angle < 0:
        angle += np.pi 
    return angle

def shift(one, another):
    return (one[0] + another[0], one[1] + another[1])

def bfs(image, components, n_bins=37):
    weights = np.zeros(n_bins, np.float64)
    segments = []

    for i in range(1, len(components)):
        visited = set()
        first = next(iter(components[i])) 
        q = [(first, first)]
        
        for cur, prev in q:
            visited.add(cur)
            neighbors = 0
            
            appending_new = []
            cur_dist = distance(cur, prev)
            for step in steps:
                new = shift(cur, step) 
                if (new in components[i]) and (new not in visited):
                    neighbors += 1
                    appending_new.append(new)
            
            if cur_dist < segment_length:
                appending_prev = len(appending_new) * [prev]
            else:
                appending_prev = len(appending_new) * [cur]
            q += zip(appending_new, appending_prev)
            
            if (neighbors == 0) or ((cur_dist >= segment_length) and (neighbors == 1)):
                bin_number = int(np.round((angle(cur, prev) * 180 / np.pi) / 5))
                weights[bin_number] += cur_dist

    return weights

steps = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
segment_length = 10

def get_statistics(image, n_bins=38):
    skel = medial_axis(image)
    blobs_labels = label(skel, background=0)
    
    image_h = processed_image.shape[0]
    image_w = processed_image.shape[1]
    
    coords = [(i, j) for i in range(image_h) for j in range(image_w)]
    points_mapping = sorted(zip(blobs_labels.flatten(), coords), key=lambda arg: arg[0])
    groups = itertools.groupby(points_mapping, key=lambda arg: arg[0])
    components = [{item[1] for item in group} for key, group in groups]
    
    weights = bfs(blobs_labels, components)
    area = np.where(np.logical_not(image == 0))[0].size
    length = weights.sum()
    return area, length, weights

