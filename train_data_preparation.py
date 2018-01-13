
# coding: utf-8

# In[1]:

import os
from itertools import product

import tqdm
import lmdb

import numpy as np

from scipy.misc import imread, imsave, imresize


# In[2]:

def labeled_img_preprocess_binary_case(image):
    mask = np.where(image < 128, np.zeros_like(image), np.ones_like(image))
    mask = mask.astype(np.int8)
    return mask


# In[3]:

def input_img_preprocess(image):
    image = image.astype(np.float32)
    image = image / 255 - 0.5
    return image


# In[4]:

def shapes_preprocess(img):
    if img.ndim == 2:
        img = img.reshape(img.shape + (1, ))
    img = img.reshape((1, ) + img.shape) 
    img = img.transpose(0, 3, 1, 2)
    return img


# In[5]:

def get_valid_patches(img_shape, patch_size, central_points):
    start = central_points - patch_size / 2
    end = start + patch_size
    
    mask = np.logical_and(start >= 0, end < np.array(img_shape))
    mask = np.all(mask, axis=-1)
    
    return mask


def extract_grid_patches(input_img, labeled_img, patch_size=64, h_patches_number=10, w_patches_number=10):
    H = input_img.shape[0]
    W = input_img.shape[1]
    
    X = []
    Y = []
    
    hes = np.linspace(patch_size // 2, H - (patch_size - patch_size // 2), h_patches_number, dtype=np.int32)
    wes = np.linspace(patch_size // 2, W - (patch_size - patch_size // 2), w_patches_number, dtype=np.int32)
    positions = list(product(hes, wes))
    #np.random.shuffle(positions)

    for position in positions:
        start = np.array(position) - patch_size // 2
        end = start + patch_size
    
        x = shapes_preprocess(
            input_img[start[0]:end[0], start[1]:end[1]]
        )
        y = shapes_preprocess(
            labeled_img[start[0]:end[0], start[1]:end[1]]
        )
        X.append(x)
        Y.append(y)

    return X, Y


def extract_random_patches(input_img, labeled_img, n_classes=2, patch_size=64, class_patches_number=100):
    X = []
    Y = []
    
    for label in range(n_classes):
        positions = np.argwhere(labeled_img == label)
        
        accepted_patches_mask = get_valid_patches(labeled_img.shape, patch_size, positions)
        positions = positions[accepted_patches_mask][:class_patches_number]
        np.random.shuffle(positions)
        
        for position in positions:
            start = position - patch_size / 2
            end = start + patch_size
            
            x = shapes_preprocess(
                input_img[start[0]:end[0], start[1]:end[1]]
            )
            y = shapes_preprocess(
                labeled_img[start[0]:end[0], start[1]:end[1]]
            )
            X.append(x)
            Y.append(y)
        
    return X, Y


# In[6]:

def get_imgs_from_folder(fpath, proportion):
    stack = []
    size = len(os.listdir(fpath))
    for i, fname in enumerate(sorted(os.listdir(fpath))):
        img = imread(os.path.join(fpath, fname))
        stack.append(img)
        if 1.0 * i / size > proportion:
            break
    return stack


# In[7]:

def get_data(path_X, path_Y, patch_extraction_function, patch_extraction_params, proportion):
    X = get_imgs_from_folder(path_X, proportion)
    Y = get_imgs_from_folder(path_Y, proportion)
    
    patches_X, patches_Y = [], []
    for x, y in tqdm.tqdm_notebook(zip(X, Y)):
        x = input_img_preprocess(x)
        y = labeled_img_preprocess_binary_case(y)
        subpatches_X, subpatches_Y = patch_extraction_function(x, y, **patch_extraction_params)
        patches_X += subpatches_X
        patches_Y += subpatches_Y
        
    X = np.concatenate(patches_X)
    Y = np.concatenate(patches_Y)
        
    Y = Y.reshape(Y.shape[0], -1)
    return X, Y

  
def fill_storage(name, data):
    N = data.shape[0]
    env = lmdb.open(name, map_size=1e+10)
    with env.begin(write=True) as txn:
        start = env.stat()["entries"]
        for i in range(start, start + N):
            str_id = '{:06}'.format(i)
            txn.put(str_id.encode('ascii'), data[i - start].tobytes())
    env.close()

            
def prepare_data(path_X, path_Y, random_state, patch_extraction_function, patch_extraction_params, proportion):
    np.random.seed(random_state)
    
    X, Y = get_data(path_X, path_Y, patch_extraction_function, patch_extraction_params, proportion)
    
    fill_storage('../output_data/storage/input_images.lmdb', X)
    fill_storage('../output_data/storage/labeled_images.lmdb', Y)