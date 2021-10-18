import keras.backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
import numpy as np

def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3,4]) ##y_true与y_pred都是矩阵！（Unet）
    union = K.sum(y_true, axis=[1,2,3,4]) + K.sum(y_pred, axis=[1,2,3,4])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_2d(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1,2,3]) ##y_true与y_pred都是矩阵！（Unet）
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
def compute_binary_dice(input1, input2):
    mask1 = input1
    mask2 = input2
    vol1 = tf.reduce_sum(tf.to_float(mask1), axis=[1, 2, 3, 4])
    vol2 = tf.reduce_sum(tf.to_float(mask2), axis=[1, 2, 3, 4])
    dice = tf.reduce_sum(tf.to_float(mask1 & mask2), axis=[1, 2, 3, 4])*2 / (vol1+vol2)
    return dice
def dice_coef1(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred)  ##y_true与y_pred都是矩阵！（Unet）
    union = K.sum(y_true) + K.sum(y_pred)
    return K.mean((2. * intersection + smooth) / (union + smooth))
def dice_p_bce(in_gt, in_pred):
    return 1e-3*binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred, smooth=1)
def dice_coef_loss_2d(y_true, y_pred):
    return 1 - dice_coef_2d(y_true, y_pred, smooth=1)
'''
Compatible with tensorflow backend
'''
def ssim(ts,ps):
    ssim_v = tf.reduce_sum(tf.image.ssim(ts, ps, 1.0))
    return ssim_v


def focal_loss(y_true, y_pred):
    gamma = 0.75
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_0 = K.clip(pt_0, 1e-3, .999)

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum(
        (1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

def generalized_dice_coeff(y_true, y_pred):
    '''
    https://arxiv.org/pdf/1707.03237.pdf
    '''
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2,3))  # Count the number of pixels in the target area
    w = 1/(w**2+0.000001) # Calculate category weights
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3,4))
    numerator = K.sum(numerator)  #molecular


    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3,4))
    denominator = K.sum(denominator)  #denominator

    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)

def generalized_dice_coeff_2d(y_true, y_pred):
    '''
    https://arxiv.org/pdf/1707.03237.pdf
    '''
    Ncl = y_pred.shape[-1]
    w = K.zeros(shape=(Ncl,))
    w = K.sum(y_true, axis=(0,1,2))  # Count the number of pixels in the target area
    w = 1/(w**2+0.000001) # Calculate category weights
    # Compute gen dice coef:
    numerator = y_true*y_pred
    numerator = w*K.sum(numerator,(0,1,2,3))
    numerator = K.sum(numerator)  #molecular


    denominator = y_true+y_pred
    denominator = w*K.sum(denominator,(0,1,2,3))
    denominator = K.sum(denominator)  #denominator

    gen_dice_coef = 2*numerator/denominator
    return gen_dice_coef

def generalized_dice_loss_2d(y_true, y_pred):
    return 1 - generalized_dice_coeff_2d(y_true, y_pred)


def IoU(y_true, y_pred, eps=1e-6):
    # if np.max(y_true) == 0.0:
    #     return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3,4])
    union = K.sum(y_true, axis=[1,2,3,4]) + K.sum(y_pred, axis=[1,2,3,4]) - intersection
    return K.mean( (intersection + eps) / (union + eps), axis=0)

def iou_loss(in_gt, in_pred):
    return - IoU(in_gt, in_pred)


def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    smooth = 0.3
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)


def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)
def my_loss(y_true, y_pred):
    loss1 = K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)
    loss2 = K.mean(K.square(y_pred - y_true), axis=-1)
    return loss1+loss2
def mean_squared_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.square(y_pred - y_true), axis=-1)

def my_loss1(y_true, y_pred):
    loss1 =  K.categorical_crossentropy(y_pred , y_true)
    loss3 = generalized_dice_loss(y_true, y_pred)
    return loss1+loss3
def my_loss_mutil_out(y_true, y_pred):
    y_true = y_true[...,1:4]
    y_pred = y_pred[...,1:4]
    loss1 =  K.categorical_crossentropy(y_pred , y_true)
    loss3 = generalized_dice_loss(y_true, y_pred)
    return loss1+loss3
def my_loss1_2d(y_true, y_pred):
    loss1 =  K.categorical_crossentropy(y_pred , y_true)
    loss2 = focal_loss(y_true, y_pred)
    loss3 = generalized_dice_loss_2d(y_true, y_pred)
    loss4 = K.mean(K.square(y_pred - y_true), axis=-1)
    return loss1+loss3

def csf_loss(y_true, y_pred):
    loss1 = K.binary_crossentropy(y_true, y_pred)
    loss2 = focal_loss(y_true, y_pred)
    loss3 = dice_coef_loss(y_true, y_pred)
    loss4 = K.mean(K.square(y_pred - y_true), axis=-1)
    return loss1+loss3

def csf_loss_2d(y_true, y_pred):
    loss1 = K.binary_crossentropy(y_true, y_pred)
    loss2 = focal_loss(y_true, y_pred)
    loss3 = dice_coef_loss_2d(y_true, y_pred)
    loss4 = K.mean(K.square(y_pred - y_true), axis=-1)
    return loss1+loss3


def get_reference_grid(grid_size):
    return tf.to_float(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3))
def compute_centroid_distance(input1, input2, grid=None):
    if grid is None:
        grid = get_reference_grid(input1.get_shape()[1:4])

    def compute_centroid(mask, grid0):
        return tf.stack([tf.reduce_mean(tf.boolean_mask(grid0, mask[i, ..., 0] >= 0.5), axis=0)
                         for i in range(mask.shape[0].value)], axis=0)
    c1 = compute_centroid(input1, grid)
    c2 = compute_centroid(input2, grid)
    return tf.sqrt(tf.reduce_sum(tf.square(c1-c2), axis=1))

def dice(vol1, vol2, labels=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    labels : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed
    '''
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background

    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    if nargout == 1:
        return dicem
    else:
        return (dicem, labels)
