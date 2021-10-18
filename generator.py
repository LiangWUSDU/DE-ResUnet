import numpy as np
from data_gen import *
import fast_glcm
import nibabel as nib


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.flip(image, axis=(1))
        mask = np.flip(mask, axis=(1))
    return image, mask

def randomVerticalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.flip(image, axis=(2))
        mask = np.flip(mask, axis=(2))
    return image, mask

def randomRotation(image, mask, u=0.5):
    n_rand = np.random.random()
    if n_rand < u:
        if n_rand <= u/3:
            k=1
        elif n_rand > u - u/3:
            k=3
        else:
            k=2
        image = np.rot90(image, k=k, axes=(1,2))
        mask = np.rot90(mask, k=k, axes=(1,2))
    return image, mask
def argument(image,mask,axis):
    image = np.flip(image,axis=axis)
    mask = np.flip(mask,axis=axis)
    return image,mask
def generator_patch(x,y):
    x1,x2,x3,x4 = GLCM(x,192)
    train_patch = vols_generator_patch(vol_name=x, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=1, num_images=80)
    train_patch1 = vols_generator_patch(vol_name=x1, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=1, num_images=80)
    train_patch2 = vols_generator_patch(vol_name=x2, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=1, num_images=80)
    train_patch3 = vols_generator_patch(vol_name=x3, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=1, num_images=80)
    train_patch4 = vols_generator_patch(vol_name=x4, num_data=1, patch_size=[64,64,64],
                                          stride_patch=[32,32,32], out=1, num_images=80)
    mask_patch = vols_mask_generator_patch(vol_name=y, num_data=1, patch_size=[64,64,64,4],
                                                     stride_patch=[32,32,32,4], out=1, num_images=80)
    return train_patch,train_patch1,train_patch2,train_patch3,train_patch4,mask_patch

def random(x_train,x_mask):
    permutation = np.random.permutation(x_mask.shape[0])
    x_train = x_train[permutation, :, :]
    x_mask = x_mask[permutation]
    return x_train,x_mask
def randomm(x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch):
    permutation = np.random.permutation(x_patch.shape[0])
    x_patch = x_patch[permutation, :, :]
    x_patch1 = x_patch1[permutation, :, :]
    x_patch2 = x_patch2[permutation, :, :]
    x_patch3 = x_patch3[permutation, :, :]
    x_patch4 = x_patch4[permutation, :, :]
    y_patch = y_patch[permutation]
    return x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch
def GLCM(image,num):
    image = np.reshape(image, (160,192,160))
    x1 = np.ones((1,160,192,160))
    x2 = np.ones((1,160,192,160))
    x3 =np.ones((1,160,192,160))
    x4 = np.ones((1,160,192,160))
    for i in range(num):
        img = image[:,i,:]
        img = np.reshape(img,(160,160))
        img = img*255
        mean = fast_glcm.fast_glcm_mean(img)
        std = fast_glcm.fast_glcm_std(img)
        cont = fast_glcm.fast_glcm_contrast(img)
        diss = fast_glcm.fast_glcm_dissimilarity(img)
        mean = mean/255
        std = std/255
        cont = cont/255
        diss =diss/255
        x1[:,:,i,:]=mean
        x2[:,:,i,:]=std
        x3[:,:,i,:]=cont
        x4[:,:,i,:]=diss
    return x1,x2,x3,x4
def generator_data(image_file,st,label_file,sl):
    vol_dir = image_file + st
    image1 = nib.load(vol_dir)
    image = image1.get_data()
    affine0 = image1.affine.copy()
    image = np.asarray(image, dtype=np.float32)
    image = np.reshape(image,(160,192,160))
    label_dir = label_file+sl
    label = nib.load(label_dir).get_data()
    label = np.asarray(label, dtype=np.float32)
    affine0 = np.asarray(affine0, dtype=np.float32)
    return image,label,affine0
def sig2(label):
    label = np.reshape(label,(160,192,160))
    BK = np.zeros((160,192,160))
    CSF = np.zeros((160,192,160))
    GM = np.zeros((160,192,160))
    WM = np.zeros((160,192,160))
    BK[label ==0] =1
    CSF[label==1] =1
    GM[label ==2] =1
    WM[label ==3] =1
    BK = BK[...,np.newaxis]
    CSF = CSF[...,np.newaxis]
    GM = GM[...,np.newaxis]
    WM = WM[...,np.newaxis]
    labels = np.concatenate([BK,CSF,GM,WM],axis= -1)
    return labels

def BatchGenerator(trainX,trainY, batch_size,batch_patch_size, augment = True):
    while True:
        for start in range(0, len(trainX), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(trainX))
            for id in range(start,end):
                img = trainX[id]
                mask = trainY[id]
                mask = sig2(mask)
                # augment data by random horizontal and vertical flip
                if augment:
                   img, mask = argument(img,mask,axis= 0)
                   img, mask = argument(img,mask,axis= 1)
                   img, mask = argument(img,mask,axis= 2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch = generator_patch(x_batch,y_batch)
            for start in range(0, len(x_patch), batch_patch_size):
                x_batch_patch = []
                x_batch_patch1 = []
                x_batch_patch2 = []
                x_batch_patch3 = []
                x_batch_patch4 = []
                y_batch_patch = []
                end = min(start + batch_patch_size, len(x_patch))
                for id in range(start, end):
                    img_patch = x_patch[id]
                    img_patch1 = x_patch1[id]
                    img_patch2 = x_patch2[id]
                    img_patch3 = x_patch3[id]
                    img_patch4 = x_patch4[id]
                    mask_patch = y_patch[id]
                    x_batch_patch.append(img_patch)
                    x_batch_patch1.append(img_patch1)
                    x_batch_patch2.append(img_patch2)
                    x_batch_patch3.append(img_patch3)
                    x_batch_patch4.append(img_patch4)
                    y_batch_patch.append(mask_patch)
                x_batch_patch = np.array(x_batch_patch)
                x_batch_patch1 = np.array(x_batch_patch1)
                x_batch_patch2 = np.array(x_batch_patch2)
                x_batch_patch3 = np.array(x_batch_patch3)
                x_batch_patch4 = np.array(x_batch_patch4)
                y_batch_patch = np.array(y_batch_patch)
                x_batch_patch = x_batch_patch[...,np.newaxis]
                x_batch_patch1 = x_batch_patch1[...,np.newaxis]
                x_batch_patch2 = x_batch_patch2[...,np.newaxis]
                x_batch_patch3 = x_batch_patch3[...,np.newaxis]
                x_batch_patch4 = x_batch_patch4[...,np.newaxis]
                y_CSF = y_batch_patch[..., 1:2]
                y_GM = y_batch_patch[..., 2:3]
                y_WM = y_batch_patch[..., 3:4]
                yield ([x_batch_patch, x_batch_patch1, x_batch_patch2, x_batch_patch3, x_batch_patch4],
                       {'output1': y_batch_patch, 'output': y_batch_patch, 'CSF': y_CSF, 'GM': y_GM, 'WM': y_WM})

def BatchGenerator2(trainX,trainY, batch_size,batch_patch_size, augment = True):
    while True:
        for start in range(0, len(trainX), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(trainX))
            for id in range(start,end):
                img = trainX[id]
                mask = trainY[id]
                mask = sig2(mask)
                # augment data by random horizontal and vertical flip
                if augment:
                   img, mask = argument(img,mask,axis= 0)
                   img, mask = randomRotation(img,mask)
                   img, mask = argument(img,mask,axis= 2)
                   img, mask = argument(img,mask,axis= 1)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)
            x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch = generator_patch(x_batch,y_batch)
            x_patch, x_patch1, x_patch2, x_patch3, x_patch4, y_patch = randomm(x_patch,x_patch1,x_patch2,x_patch3,x_patch4,y_patch)
            for start in range(0, len(x_patch), batch_patch_size):
                x_batch_patch = []
                x_batch_patch1 = []
                x_batch_patch2 = []
                x_batch_patch3 = []
                x_batch_patch4 = []
                y_batch_patch = []
                end = min(start + batch_patch_size, len(x_patch))
                for id in range(start, end):
                    img_patch = x_patch[id]
                    img_patch1 = x_patch1[id]
                    img_patch2 = x_patch2[id]
                    img_patch3 = x_patch3[id]
                    img_patch4 = x_patch4[id]
                    mask_patch = y_patch[id]
                    x_batch_patch.append(img_patch)
                    x_batch_patch1.append(img_patch1)
                    x_batch_patch2.append(img_patch2)
                    x_batch_patch3.append(img_patch3)
                    x_batch_patch4.append(img_patch4)
                    y_batch_patch.append(mask_patch)
                x_batch_patch = np.array(x_batch_patch)
                x_batch_patch1 = np.array(x_batch_patch1)
                x_batch_patch2 = np.array(x_batch_patch2)
                x_batch_patch3 = np.array(x_batch_patch3)
                x_batch_patch4 = np.array(x_batch_patch4)
                y_batch_patch = np.array(y_batch_patch)
                x_batch_patch = x_batch_patch[...,np.newaxis]
                x_batch_patch1 = x_batch_patch1[...,np.newaxis]
                x_batch_patch2 = x_batch_patch2[...,np.newaxis]
                x_batch_patch3 = x_batch_patch3[...,np.newaxis]
                x_batch_patch4 = x_batch_patch4[...,np.newaxis]
                y_CSF = y_batch_patch[..., 1:2]
                y_GM = y_batch_patch[..., 2:3]
                y_WM = y_batch_patch[..., 3:4]
                yield ([x_batch_patch, x_batch_patch1, x_batch_patch2, x_batch_patch3, x_batch_patch4],
                       {'output8': y_batch_patch,'output16': y_batch_patch,'output32': y_batch_patch,'output64': y_batch_patch,'output1': y_batch_patch, 'output2': y_batch_patch,
                        'CSF': y_CSF, 'GM': y_GM, 'WM': y_WM})