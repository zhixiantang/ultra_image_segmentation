#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev
"""

from __future__ import division, print_function
import numpy as np

import dicom
from collections import defaultdict
import os, pickle, sys
import shutil
import matplotlib.pyplot as plt
import nrrd
import scipy.io
from skimage.transform import resize
from skimage.exposure import equalize_adapthist, equalize_hist


from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint


from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator


from models import actual_unet, simple_unet
from metrics import dice_coef, dice_coef_loss, numpy_dice

from augmenters import *

def dicom_to_array(img_rows, img_cols):


    TrainData_dict = scipy.io.loadmat('../data/train.mat')
    TrainMask_dict = scipy.io.loadmat('../data/trainseg.mat')
    ValidateData_dict = scipy.io.loadmat('../data/validate.mat')
    ValidateMask_dict = scipy.io.loadmat('../data/validateseg.mat')
    TestData_dict = scipy.io.loadmat('../data/test.mat')
    TestMask_dict = scipy.io.loadmat('../data/testseg.mat')

    TrainData = TrainData_dict['image_all']
    TrainMask = TrainMask_dict['seg_all']
    ValidateData = ValidateData_dict['image_all']
    ValidateMask = ValidateMask_dict['seg_all']
    TestData = TestData_dict['image_all']
    TestMask = TestMask_dict['seg_all']


    d1 = TrainData.shape[0]
    d2 = ValidateData.shape[0]   
    d3 = TestData.shape[0]

    imgs1 = np.zeros( [d1, img_rows, img_cols])
    imgs1_seg = np.zeros( [d1, img_rows, img_cols])
    imgs2 = np.zeros( [d2, img_rows, img_cols])
    imgs2_seg = np.zeros( [d2, img_rows, img_cols])
    imgs3 = np.zeros( [d3, img_rows, img_cols])
    imgs3_seg = np.zeros( [d3, img_rows, img_cols])


    for kk in range(0,d1):  #for train
        img = TrainData[kk,:,:]
        TrainData[kk,:,:] = equalize_hist( img.astype(int) )
        imgs1[kk,:,:] = resize( TrainData[kk,:,:], (img_rows, img_cols), preserve_range=True)
        imgs1_seg[kk,:,:] = resize( TrainMask[kk,:,:], (img_rows, img_cols), preserve_range=True)

    for kk in range(0,d2):  #for validate
        img = ValidateData[kk,:,:]
        ValidateData[kk,:,:] = equalize_hist( img.astype(int) )
        imgs2[kk,:,:] = resize( ValidateData[kk,:,:], (img_rows, img_cols), preserve_range=True)
        imgs2_seg[kk,:,:] = resize( ValidateMask[kk,:,:], (img_rows, img_cols), preserve_range=True)

    for kk in range(0,d3):  #for test
        img = TestData[kk,:,:]
        TestData[kk,:,:] = equalize_hist( img.astype(int) )
        imgs3[kk,:,:] = resize( TestData[kk,:,:], (img_rows, img_cols), preserve_range=True)
        imgs3_seg[kk,:,:] = resize( TestMask[kk,:,:], (img_rows, img_cols), preserve_range=True)


    TrainData = np.concatenate(imgs1, axis=0).reshape(-1, img_rows, img_cols, 1)
    TrainMask = np.concatenate(imgs1_seg, axis=0).reshape(-1, img_rows, img_cols, 1)
    ValidateData = np.concatenate(imgs2, axis=0).reshape(-1, img_rows, img_cols, 1)
    ValidateMask = np.concatenate(imgs2_seg, axis=0).reshape(-1, img_rows, img_cols, 1)
    TestData = np.concatenate(imgs3, axis=0).reshape(-1, img_rows, img_cols, 1)
    TestMask = np.concatenate(imgs3_seg, axis=0).reshape(-1, img_rows, img_cols, 1)

    np.save('../data/train.npy', TrainData)
    np.save('../data/train_masks.npy', TrainMask)
    np.save('../data/validate.npy', ValidateData)
    np.save('../data/validate_masks.npy', ValidateMask)
    np.save('../data/test.npy', TestData)
    np.save('../data/test_masks.npy', TestMask)

def load_data():

    X_train = np.load('../data/train.npy')
    y_train = np.load('../data/train_masks.npy')
    X_test = np.load('../data/test.npy')
    y_test = np.load('../data/test_masks.npy')
    X_val = np.load('../data/validate.npy')
    y_val = np.load('../data/validate_masks.npy')

    return X_train, y_train, X_test, y_test, X_val, y_val

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.5
    epochs_drop = 5
    lrate = initial_lrate * drop**int((1 + epoch) / epochs_drop)
    return lrate

def keras_fit_generator(img_rows=256, img_cols=256, n_imgs=10**4, batch_size=32, regenerate=True): ##True gaichengle false

    if regenerate:
        dicom_to_array(img_rows, img_cols)
        #preprocess_data()

    X_train, y_train, X_test, y_test, X_val, y_val = load_data()

    img_rows = X_train.shape[1]
    img_cols = img_rows

    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=90.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2)#,
        #preprocessing_function=elastic_transform)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    image_datagen.fit(X_train,seed=seed)
    mask_datagen.fit(y_train, seed=seed)

    image_generator = image_datagen.flow(X_train, batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow(y_train, batch_size=batch_size, seed=seed)

    #from itertools import izip
    train_generator = zip(image_generator, mask_generator)


    model = actual_unet( img_rows, img_cols)
    #model.load_weights('../data/weights.h5')

    model.summary()
    model_checkpoint = ModelCheckpoint(
        '../data/weights.h5', monitor='val_loss', save_best_only=True)

    lrate = LearningRateScheduler(step_decay)

    model.compile(  optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef, 'binary_accuracy'])

    model.fit_generator(
                        train_generator,
                        steps_per_epoch=n_imgs//batch_size,
                        epochs=30,
                        verbose=1,
                        shuffle=True,
                        validation_data=(X_val, y_val),
                        callbacks=[model_checkpoint, lrate],
                        use_multiprocessing=True)

    score = model.evaluate(X_test, y_test, verbose=2)

    print()
    print('Test accuracy:', score[1])

import time

start = time.time()
keras_fit_generator(img_rows=256, img_cols=256, regenerate=True,
                   n_imgs=15*10**4, batch_size=32)

##n_imgs=15*10**4 gaichengle 10**4

end = time.time()

print('Elapsed time:', round((end-start)/60,2 ) )
