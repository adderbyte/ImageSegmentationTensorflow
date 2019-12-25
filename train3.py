#!/usr/bin/env python

import os
import sys
from PIL import Image

import numpy as np
import tensorflow as tf
import  yaml
import imageProcessor
import _pickle as pkl

import skimage
import random
import model 




class TensorFlowTrainer(object):
    """

       This function is the control center for training
       It uses the imageProcessor module and the model module
       to achieve this functionality.
       Note that the config file is very important since it
       provides a central location for our parameters.

       Check the segmentation.ipynb for getting started on how to use 
       the modules.

       @author Lukman Olagoke Olabisi



    """

    def __init__(self, trainValLoss='trainValloss',testoutput= 'segTest'):

        '''
        Initialise all necessary modules
        input : 
                trainVallos: name for folder for storing train loss 
                testoutput: name of folder storing output of compuation
                validation: if validation should be performed

        '''
        # the seeion
        self.session = None
        # current directory
        self.workPath = os.getcwd()
        # check if config file is in current directory otherwise raise error
        self.configCheck = os.path.isfile(self.workPath + '/config.yml') 
        # get the config file
        self.config = self.__configFile()
        # check data path
        self.__dataPath()
        # load data
        self.__dataLoad()
        # set train batch size . this can be changed when model is launched
        self.train_batch_size = 3
        # save path for storing checkpoints of model
        self.savepath = self.workPath  + self.config['model']['save']
        # output of model
        self.output = self.workPath  + self.config['model']['output']
        #self.tensorboard = self.workPath  + self.config['model']['logpath']
        # test data file name, needed for pickling losses
        self.testoutputfileName = testoutput
        # validation data file name
        self.trainValloss = trainValLoss
        # create a session
        self.session=tf.Session() 
        

        ################################################################################
        # check folders for saving model exist
        # create validation , train and test data
        # run train set up
        self.__trainSetup()
        ################################################################################

        ################################################################################
        # initialise model parameters
        # note that it is this module that will do the prediction and optimization
        self.model = model.Model(self.training_size,self.img_patch_size,self.config['model'],channels=np.array(self.data[0][0]).ndim,sess = self.session)
        ################################################################################
    


        

    def __dataPath(self):
        '''
        checks thats the data path is properly configured
        input: nil
        output : nil
        '''

        # ################################################################################
        # makes use of the keys in config file  and test that the 
        # necessaary folders are there otherwise raise exception
        if 'datapath'  in self.config.keys():
            if 'imageTrainDir' and 'imageTargetDir' and 'ImageTargetPairPickled' in self.config['datapath'].keys():
                self.data_dir = self.workPath+self.config['datapath']['ImageTargetPairPickled']
            else:
                raise Exception('ImageTargetPairPickled,imageTargetDir,imageTrainDir keys are missing in config file. imageTrainDir is path to train Images. imageTargetDir is path to target images. ImageTargetPairPickled is path for storing preprocessed images')
        # if  r format of data directory not as expected raise exception above
        ################################################################################
        else:
            ################################################################################
            # if no config file found with data dir found
            raise Exception('datapath not set in the config file')
            ################################################################################

    def __dataLoad(self):
        '''
        INput: nil
        output: prepares the data set. check data is available in the data directory
                and loads it; otherwise it runs the imageProcessor function

        '''
        ################################################################################
        # get the folders in data direc
        # check if pickle file present for data
        list_dir = os.listdir(self.data_dir)
        if list_dir :
            for name in list_dir:
                with open(self.data_dir+name, "rb") as input_file:
                     self.data = pkl.load(input_file) 
            print('Total data loaded :',len(self.data))
        ################################################################################
        else:
        ################################################################################
        # if no data found preprocess the data here   
            print('No preprocessed pickle data file dound. System is helping on this ...')
            imageProcessor.main()
            list_dir = os.listdir(self.data_dir)
            if list_dir:
                for name in list_dir:
                    with open(self.data_dir+name, "rb") as input_file:
                         self.data = pkl.load(input_file)
        ################################################################################
            else:
                # raise exception if no data folder and if preprocessing cant be done
                raise  Exception('Data cannot be loaded or preprocessed. Check config file properly')
        ################################################################################

    def __configFile(self):
        '''
        This function enables to check for config file
        Add as many agent as you want here.
        '''
       ################################################################################
        # Load the config file 
        #print(self.configCheck)
        if self.configCheck:
            with open("config.yml", 'r') as ymlfile:
                    config = yaml.safe_load(ymlfile)
            return config
       ################################################################################
        else:
            raise Exception('Config file Missing !')


    def __set_training_size(self):
        '''
        inout: nil
        output: training size 
        '''
        ################################################################################
        self.training_size = list(skimage.img_as_float(self.train[0][0]).shape)[:2]
        ################################################################################
    def __set_img_patch_size(self):
        '''
        INPUT : target segment size
        output: target size
        '''
        ################################################################################
        self.img_patch_size =  list(skimage.img_as_float(self.train[0][1]).shape)
        ################################################################################

    @staticmethod
    def image_to_arrays(img):
        '''
        input : the image
        output: conveted image to array
        I used skimage because the output is consistent for .png images
        with PIL package
        train and test conversion function function
        '''
        ################################################################################
        image = skimage.img_as_float(img)
        ################################################################################
        return image

    @staticmethod
    def imageSeg(img):
        '''
        input : conver array to gray scale 

        '''
        ################################################################################
        img = Image.fromarray(np.uint8(img * 255) , 'L')
        img.show()
        ################################################################################


    def __trainValidationTest(self):
        '''
        input : nil
        output:
                validation set, train set and testset for data

        '''
        #
        ################################################################################
        # data segmentation
        # the partition is based on the configuratoon ratio  in the config file
        self.test = self.data[:self.config['model']['TestSize']]
        trainDatas = self.data[self.config['model']['TestSize']:]
        self.validationData = trainDatas[:self.config['model']['validationSize']]
        self.train = trainDatas[self.config['model']['validationSize']:] 
       ################################################################################
        


    def __get_prediction(self):
        '''
        input : nil

        output: segmentation prediction 

        '''
        self.model.saver.restore(self.session, self.savepath + "model.ckpt")

        ################################################################################
        # convert image to array
        converter = lambda x: TensorFlowTrainer.image_to_arrays(x)
        # store output in the list below
        self.collector=[]
        losses= []
        ################################################################################
        
        ################################################################################
        # the parameters for testing
        # note here that we select randomly from the test data set 
        # according to thebatch size
        batch_sizes =3
        indices = range(len(self.test))
        perm_indices = list(np.random.permutation(indices))
        chosen_indices = random.choices(perm_indices, k=batch_sizes)
        dataVal = [self.test[i] for i in chosen_indices]
        ################################################################################

        ################################################################################
        # convert data to array
        test_data = [converter(i[0]) for i in dataVal] #self.train_data[batch_indices, :, :, :]
        test_labels = [converter(i[1]) for i in dataVal] #self.train_labels[batch_indices]
        ################################################################################

            
        ################################################################################
        # run model
        # note that one needs to set restore to true so that we can test
        # using trained model
        Vallosses= self.session.run(
                    [self.model.runModel(test_data,test_labels,train=False ,batch=3)])
        ################################################################################
        
        ################################################################################
        #collect results here
        self.collector.append(Vallosses)
        ################################################################################     

        ################################################################################
        # if we intend tp save the sesults as pickle file
        # with open(self.testoutputfileName+'.pkl', 'wb') as f:
        #     pkl.dump(zip(collector,losses), f)
        ################################################################################
    

    def  __trainSetup(self):
        '''
        INPUT: NIL

        This prepares the model for training
        doing the necessary preparation before launch

        '''
        print('starting data train/test separation ...')

        ################################################################################
        # create folder for saving model if this does not exist
        imageProcessor.modelFolderCreate(self.workPath);
        ################################################################################
        # this partition the data set see above 
        self.__trainValidationTest();
        ################################################################################

        ################################################################################
        self.__set_training_size(); self.__set_img_patch_size(); # set training and test size
        ################################################################################
        
        print('completed separation')



    def predict(self, num_epochs=3, restore=False, test=False,validation=False):
        '''
        input: 
                num_epochs: the number of training epoch
                restore: False , set to true for testing model

        '''

        
        
        ################################################################################
        # use lambda expression to facilitate image conversion
        # this convert image to array
        converter = lambda x: TensorFlowTrainer.image_to_arrays(x)
        ################################################################################
       
       ################################################################################
       # this is for storing train, test and validation loss
        self.validations = validation
        self.validationloss = []
        self.trainloss = []
       ################################################################################
        

      ################################################################################
      # launch a tensorflow session





        ################################################################################
        # thefirst statement here ensures we only test if there is a module stored already
        # run test
        if restore and test:
            # Restore variables from disk.

            
            print("Model restored (restore set to true.")
            print('starting model testing')
            #self.saver.restore(s, self.savepath + "model.ckpt")
            # get prediction module help us do the prediction
            self.__get_prediction()
            #restore=False
        ################################################################################
        else:
            ################################################################################
            # for training and validation
            print('Training model ...')
            if restore:
                self.model.saver.restore(self.session, self.savepath + "model.ckpt")
            # retore model if you want to continue training
            ################################################################################

            ################################################################################
            # run through the data set
            # pick randomly from it when the data set has been exhausted fully
            training_indices = range(self.training_size[0])
            start= 0 
            stop = self.train_batch_size
            print('Strating Lauching of the segmentation model...')
            ################################################################################

            ################################################################################
            # trianinf loop
            for iepoch in range(num_epochs):
                ################################################################################
                # go through full data set
                if stop < len(self.train):
                    data  = self.train[start:stop]#np.random.permutation(self.train_data)
                    start += self.train_batch_size
                    stop  += self.train_batch_size
                ################################################################################
                else:
                ################################################################################
                # once the data is completely iterated 
                # select next batch randomly
                    indices = range(len(self.train))
                    perm_indices = list(np.random.permutation(indices))
                    chosen_indices = random.choices(perm_indices, k=self.train_batch_size)
                    data = [self.train[i] for i in chosen_indices]
                ################################################################################
                
                ################################################################################
                # convert the next batch to array
                batch_data = [converter(i[0]) for i in data] #self.train_data[batch_indices, :, :, :]
                batch_labels = [converter(i[1]) for i in data] #self.train_labels[batch_indices]
                ################################################################################
                
                ################################################################################
                # stack training batch
                batch_data_train = np.stack( batch_data, axis=0 )
                batch_data_labels = np.stack( batch_labels, axis=0 )
                ################################################################################

                ################################################################################
                # now the initialise model will be used for segmentaion and traning
                trainlosses = self.session.run(
                    [self.model.runModel(batch_data_train,batch_data_labels,train=True ,batch=3)])
                # note the syntax here and the difference in parameters settings
               ################################################################################
                #print(trainlosses)
                ################################################################################
                # store train loss
                self.trainloss.append(trainlosses)
                ################################################################################
                

                ################################################################################
                # if validation
                if(self.validations):
                    print('Validating Model ...')
                    ################################################################################
                    # prepare validation data
                    validation_batch_sizes = 3
                    indices = range(len(self.validationData))
                    perm_indices = list(np.random.permutation(indices))
                    chosen_indices = random.choices(perm_indices, k=validation_batch_sizes)
                    dataVal = [self.validationData[i] for i in chosen_indices]
                    ################################################################################

                    ################################################################################
                    # convert to array
                    valid_data = [converter(i[0]) for i in dataVal] #self.train_data[batch_indices, :, :, :]
                    valid_labels = [converter(i[1]) for i in dataVal] #self.train_labels[batch_indices]
                   ################################################################################

                   ################################################################################
                   # stack data together
                    valid_data_train = np.stack( valid_data, axis=0 )
                    valid_labels = np.stack( valid_labels, axis=0 )
                   ################################################################################

                    
                   ################################################################################
                   # run validation
                   # note that train is set to false
                   # so that one wont optimize while validating
                    Vallosses= self.session.run(
                    [self.model.runModel(valid_data_train,valid_labels,train=False ,batch=3)])
                  ################################################################################
                    
                  ################################################################################
                  # save the loss
                    self.validationloss.append(Vallosses)
                  ################################################################################

                ################################################################################
                # if all goes well
                # it is time to print the tain and validation loss
                # if validation is false then it will be skipped.
                # only train loss will be reported
                if iepoch % 2 == 0 and self.validations:
                        
                    print('training_loss / validation_loss => %.2f / %.2f for step %d' % 
                        (trainlosses[0][0], Vallosses[0][0], iepoch))

                    sys.stdout.flush()
                else:

                      print('training_loss => %.2f for step %d' % 
                        (trainlosses[0][0], iepoch))
                ################################################################################


        ########### un comment to save modell ######################
        # if not test and not self.validations:
        #     with open(self.trainValloss+'.pkl', 'wb') as f:
        #             pkl.dump(zip(self.trainloss), f)

        # if not test and self.validations:
        #     with open(self.trainValloss+'.pkl', 'wb') as f:
        #             pkl.dump(zip(self.validationloss, self.Valloss), f)
        ################################################################################
            





        ################################################################################
        # gracefully close session
        #s.close()
        return self.session
        ################################################################################
            


