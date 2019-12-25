#!/usr/bin/env python

import sys
import os
import os.path
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import ImageEnhance
import skimage
import numpy as np
import random 
import _pickle as pkl
import yaml
import gc


#from matplotlib import pyplot as plt



def trapsoseFlipRotate(index,image):

    '''
    input : image
    output: transformed and resized version of the input image 
    

    Note the use of mask .
    mode of an image defines the type and depth of a pixel in the image

    '''


    ################################################################################
    # use the index to get whether it is a train or target set
    # this is important for understanding what mode to use
    train_image = True if index == 0 else False
    ################################################################################
   



    ################################################################################
    # apply image transpose and flipping
    mirror = image.transpose(Image.FLIP_LEFT_RIGHT)
    mirrorEnh = ImageEnhance.Contrast(image).enhance(1.3)

  
    return mirrorEnh
     
def modelFolderCreate(paths):
    '''
    Input : 
        paths: current directory

    Output: savedModel, log, testOutput Folders if not already existing
    '''
    ################################################################################
    # folders below are needed for saving and loading model
    # path to log folder (used for tensorboard)
    logFolder = paths + '/log'
    # path for saving model
    savedModel = paths + '/savedModel'
    # path for testOutput 
    output = paths + '/testOutput'
    #print(logFolder, savedModel, output)
    ################################################################################

    ################################################################################
    # create the directory folder above if they dont exist 
    # otherwise do nothing, raise no error
    try:
        os.makedirs(logFolder)    
    except FileExistsError:
        pass
    try:
        os.makedirs(savedModel)    
    except FileExistsError:
        pass
    try:
        os.makedirs(output)    
    except FileExistsError:
        pass
    ################################################################################



def simpleImageAugmentation(files,storepath,pickleFilename,enhance=20):

    '''
    input : 
           files:       A string .
                        file path to the train and target folder .
                        Note that the train target path are separate
                        but have been zipped into single pair

           filepath:    a string .
                        file path for the storage of the final ziped(train,target) array
                        The array of augmented train-target pair is saved as pickle file
                        into this directory



    Note : the essence of this entire function is to perform (simple data augmentation).
    This is to give us more sample data for experimentation. It is possible to use Albumentation
    library (https://github.com/albumentations-team/albumentations) t achieve similar or even more sophiscated transforms but for didactic purposes
    this transformation here are more than enough

    Note also that File paths here are loaded from the config.yml file
    '''

    ################################################################################
    # final Image stored in an array
    # images are stored as tuple of input-target pair in this array
    ImageStore = []
    ################################################################################

    

    ################################################################################
    # loop trhrough the train set and target pair folder
    # the file path are speciffied in the config.yml file
    for filepath_train,filepath_target in files:

        #print(filepath_train,filepath_target)
        try:
            # Attempt to open an image file
            # note RGB is specified here because the mode is RGBA in PIL 
            # library by default. So one need to conver to RGB
            image = Image.open(filepath_train).convert('RGB')
            # print image mode if necessary
            #print(image.mode)
            target = Image.open(filepath_target).convert('L')
            # print target  mode if necessary
            #print(target.mode)
        except (IOError):
            # Report error if file path is wrong, 
            # and then break 
            print("Problem opening image files: ", files, ":", IOError)
            break
    ################################################################################


    ################################################################################
        # put the original trian and target pair in a list
        image_target_pair = [image,target]
        # convert the list to a tuple and append to the ImageStore 
        ImageStore.append(tuple(image_target_pair))
        # define mirror and flip function using lambda expression
        # if ops is m then its mirror transformation otherwise do a flipping operation
        # ops specify the type of operation
        transMirr = lambda image,ops: ImageOps.mirror(image) if ops=='m' else ImageOps.flip(image) ;

        #define enhance and roatte function using lambda expression
        # the val is the rotation angle for rotation operation
        # it is the enhancment value for enhancement or contrast operation
        rotateEnhanceImage = lambda image,val,ops: image.rotate(val, resample=0, expand=0) if ops=='r' else ImageEnhance.Contrast(image).enhance(val)
    ################################################################################



    ################################################################################
        # mirror, flip, rotate and enhance operation performed here

        # Mirror Image using  Image target pair in the list image_target_pair and store thereafter
        imageMirror = tuple([ transMirr(i,'m')for i in image_target_pair])
        #print(skimage.img_as_float(imageMirror[0]).shape)
        ImageStore.append(imageMirror)
        # transpose Image and target pair using transMirr, ops is now f which means flip 
        imgageTransposes = tuple([transMirr(i,'f') for i in image_target_pair])
        #print(skimage.img_as_float(imgageTransposes[0]).shape)
        # save
        ImageStore.append(imgageTransposes)   
        #  to perform rotation 
        # store angles to be used in a list. this is an arbitrary choice of angle (multiple of 3)
        angles = [i for i in range(2,359) if i%3==0]
        # roate images using the anles in list above then store

        #-----------------------------------------------------------------
        ### release this later
        # image_targetPairRotated = [(rotateEnhanceImage(image,i,'r'), rotateEnhanceImage(target,i,'r')) for i in angles ]
        # ImageStore.extend(image_targetPairRotated)
        #-----------------------------------------------------------------

        # enhancement option
        # store the enhancement value in list
        enhance = [random.uniform(1.2, 1.9) for i in range(enhance)]
        # enahance the image using the enhancement value in list
        image_targetEnhanced = [(rotateEnhanceImage(image,i,'e'),rotateEnhanceImage(target,i,'e')) for i in enhance ]
        #image_targetEnhanced[0].show()
        #print(skimage.img_as_float(image_targetEnhanced[0][0]).shape)
        ImageStore.extend(image_targetEnhanced)
    ################################################################################
    
    ################################################################################
        # transpose flip mask   operation
        # this function has been defined  in this file
        # apply this function on image  and store
        image_targetTransposeFlipped = tuple([trapsoseFlipRotate(i,j) for i,j in enumerate(image_target_pair)])
        #print(skimage.img_as_float(image_targetTransposeFlipped[0]).shape)
        ImageStore.append(image_targetTransposeFlipped)
    ################################################################################

    # ################################################################################
    # save the ImageStore list file as pickle 
    # now we have the training target pair . One is ready to begin traning
    filename = pickleFilename + '.pkl'
    with open(storepath + filename, 'wb') as f:
        pkl.dump(ImageStore, f)
    
    print('Data preprocesing and storage Successful. \n Total Data size : ',len(ImageStore) )

        # clean up after the whole preprocessing. Saving some space
    del ImageStore 
    gc.collect()
    # ################################################################################
        



def main():
    ################################################################################
    # load config file
    try:
        with open("config.yml", 'r') as ymlfile:
            cfg = yaml.safe_load(ymlfile)
    except (IOError):
            print('config file is required. Put config file in current directory')
    ################################################################################

    ################################################################################
    # set and get all paths needed for uploading and storing preprocessed file
    # get working directory
    paths = os.getcwd()
    # train path
    dir_path = paths + cfg['datapath']['imageTrainDir']
    # target path
    dir_path_test = paths + cfg['datapath']['imageTargetDir']
    # get path for each file  in the train folder
    files_train = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    # get path for each file  in the target folder
    files_target = [os.path.join(dir_path_test, f) for f in os.listdir(dir_path_test) if os.path.isfile(os.path.join(dir_path_test, f))]
    # zip the train and target file path as pair. Check to ensure folder not empty
    if  files_train and    files_target:
        zippedFile = zip(files_train,files_target)
    else:
        raise Exception('missing train or test data file')
    # set the path for storing the preprocessed augmented image
    imageStoreDir = paths + cfg['datapath']['ImageTargetPairPickled']
    # pickle file name . After preprocesssing files are stored here
    pickleFilename = cfg['datapath']['pickleFilename']
    # run the image preprocessing function
    simpleImageAugmentation(zippedFile,imageStoreDir,pickleFilename)
    ################################################################################

# uncomment the lines below to run this file as stand alone

if __name__ == '__main__':
      main()