import numpy as np
import os
import glob
import sys
import random

# Third party 
import tensorflow as tf
import scipy.io as sio
from keras.utils import to_categorical

# Dr.Adrian Dalca's toolbox
from ext.pytools import patchlib
from ext.neuron  import generators
from ext.neuron  import dataproc
 
'''
data_gen.py generates the patches from the imported image. 

vols_generator_patch: generate the patches from the volumes given and store each patch in consecutive order
input - vol_name: list(files) 名字的列表
        num_data: int(number of data you want to train)  训练数据的数量
        patch_size: patch size, 3 dim  patch 的尺寸
        stride_patch: overlapping region  重叠区域
        out: out = 1 returns the list of patches in consecutive order.按顺序返回patch列表
             out = 2 returns the list of patches and the corresponding location. Unlike when out = 1,
             patches were grouped with corresponding data. 返回位置
             e.g. if num_data = 5, len(vol_patch2) = 5. len(vol_patch2[0]) = number of patches for first data

label_generator_patch: generate the patches from the labels given and store each patch in consecutive order
input - label_name: list(files)
        num_data: int(number of label data you want to train)
        patch_size: patch size, 3 dim
        stride_patch: overlapping region
        label: list of labels that you would like to use for relableling
        out: out = 1 returns the list of label patches in consecutive order.
             out = 2 returns the list of patches and the corresponding location. Unlike when out = 1
             patches were grouped with corresponding data.
             e.g. if num_data = 5, len(vol_patch2) = 5. len(vol_patch2[0]) = number of patches for first data
        
        return: list of tensors. tensor.shape = 1 + dimension of patch + number of label

'''

def vols_generator_patch (vol_name, num_data, patch_size, stride_patch, out = 1,num_images=27):

    # 120 = number of patches for one volume
    vol_patch = np.empty([num_data*num_images,64,64,64])
    #vol_patch = []
    vol_patch2 = []
    patch_loc = []
    count = 0 # count the batch size for the network

    for i in range(num_data):
        data_vol =  vol_name[i] # load the volume data from the list
        #print("volume data",i,":",vol_name[i]) # print the volume data used for training
        loc_temp = []
        temp = []
        if out == 2: 
            # generate the patch and store them in a list
            for item, loc in patchlib.patch_gen(data_vol,patch_size,stride=stride_patch, nargout = out):
                item = np.reshape(item, (1,) + item.shape + (1,))
                temp.append(item)
                loc_temp.append(loc)
            vol_patch2.append(temp)
            patch_loc.append(loc_temp)
        elif out == 1:
            for item in patchlib.patch_gen(data_vol,patch_size,stride=stride_patch):
                # vol_patch = [batch size, (dimension), channel]
                vol_patch[count,:,:,:] = item
                count+=1
                #print(count)
    if out == 1:
        return vol_patch
    elif out == 2:
        return vol_patch2, patch_loc


def re_label(label_input,num_data,labels):
    relabel = []
    for i in range(num_data):
        data_label =  label_input[i]
        data_label = generators._relabel(data_label,labels)
        relabel.append(data_label)
    return relabel

def vols_mask_generator_patch (vol_name, num_data, patch_size, stride_patch, out = 1,num_images = 27):

    # 120 = number of patches for one volume
    vol_patch = np.empty([num_data*num_images,64,64,64,4])
    #vol_patch = []
    vol_patch2 = []
    patch_loc = []
    count = 0 # count the batch size for the network

    for i in range(num_data):
        data_vol =  vol_name[i] # load the volume data from the list
        #print("volume data",i,":",vol_name[i]) # print the volume data used for training
        loc_temp = []
        temp = []
        if out == 2:
            # generate the patch and store them in a list
            for item, loc in patchlib.patch_gen(data_vol,patch_size,stride= stride_patch, nargout = out):
                item = np.reshape(item, (1,) + item.shape + (1,))
                temp.append(item)
                loc_temp.append(loc)
            vol_patch2.append(temp)
            patch_loc.append(loc_temp)
        elif out == 1:
            for item in patchlib.patch_gen(data_vol,patch_size,stride=stride_patch):
                # vol_patch = [batch size, (dimension), channel]
                vol_patch[count,:,:,:,:] = item
                count+=1
                #print(count)
    if out == 1:
        return vol_patch
    elif out == 2:
        return vol_patch2, patch_loc
def vols_mask2_generator_patch (vol_name, num_data, patch_size, stride_patch=0, out = 1,num_images = 27):

    # 120 = number of patches for one volume
    vol_patch = np.empty([num_data*num_images,64,64,64,8])
    #vol_patch = []
    vol_patch2 = []
    patch_loc = []
    count = 0 # count the batch size for the network

    for i in range(num_data):
        data_vol =  vol_name[i] # load the volume data from the list
        #print("volume data",i,":",vol_name[i]) # print the volume data used for training
        loc_temp = []
        temp = []
        if out == 2:
            # generate the patch and store them in a list
            for item, loc in patchlib.patch_gen(data_vol,patch_size,stride=[64,64,64,8], nargout = out):
                item = np.reshape(item, (1,) + item.shape + (1,))
                temp.append(item)
                loc_temp.append(loc)
            vol_patch2.append(temp)
            patch_loc.append(loc_temp)
        elif out == 1:
            for item in patchlib.patch_gen(data_vol,patch_size,stride=[64,64,64,8]):
                # vol_patch = [batch size, (dimension), channel]
                vol_patch[count,:,:,:,:] = item
                count+=1
                #print(count)
    if out == 1:
        return vol_patch
    elif out == 2:
        return vol_patch2, patch_loc
def generater_patch(train,num):
    patch_size = [128, 128, 128]
    patch_label_size = [128, 128, 128, 4]
    stride = [64, 128, 64]
    stride_label = [64, 128, 64, 4]
    y = []
    for i in range(num):
        x = train[i]
        train_vols = vols_generator_patch(vol_name=x, num_data=1, patch_size=patch_size,
                                                stride_patch=stride, out=1, num_images=9)
        y.append(train_vols)
    y= np.asarray(y,dtype=np.float32)
    return y



