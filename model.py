
import tensorflow as tf 
import pickle as pkl
import os 
import numpy as np 
from PIL import Image


class Model:
    '''
    Module that runs the segmentation training 

    '''
    def __init__(self,train_shape,target_shape,conf,channels,sess,restore=False):
        
        # the parameters needed for training 

        # the present working directory
        self.workPath = os.getcwd()
        # this contains model paramters as in config.yml file
        self.conf = conf 
        # path for folder for saving model
        self.savepath = self.workPath  + self.conf['save']
        # tensorboard log path
        self.tensorboard = self.workPath  + self.conf['logpath']
        self.seed = 66478
        # number of channels of the training image
        self.num_channels = channels
        # training features or image dimension
        self.dim1 = train_shape[0]
        self.dim2 = train_shape[1]
        # target variable dimension
        self.targetDim1 = target_shape[0]
        self.targetDim2 = target_shape[1]
        # train batch
        self.train_batch_size = None
        # this specifies the time of upsampling technique. check config.yml file
        self.upSampleOps = 'UpsampleNearNeigh'
        # initialise session == this parameter is parsed here from the traning file
        self.sess =sess
        # initialise model parameters for usage
        self.modelParams()  
        # initialise global paramters and tensorboard writer
        self.__initialised()
        # initialise saver for saving trained model
        self.saver = tf.compat.v1.train.Saver()
        
        

    def __initialised(self):
        '''
         input : Nil
         output : Nil

         The function meant to initialise variables
 
        '''
        ################################################################################
        # initialise global variables and filewriter; Funtion is called in __init__
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.writer =  tf.compat.v1.summary.FileWriter(self.tensorboard,graph= tf.compat.v1.get_default_graph() )
        ################################################################################

    def modelParams(self):
        '''
        input : Nil

        initialise the training weights and biases


        '''
        ################################################################################
        # The variables below hold all the trainable weights. They are passed an
        # initial value which will be assigned when when we call:
        # {tf.initialize_all_variables().run()}
        ################################################################################
        

        self.all_params_names = ['conv1_weights', 'conv1_biases', 'conv2_weights', 'conv2_biases', 'conv3_weights','conv3_biases',
                                'conv4_biases', 'conv4_weights', 'conv4_biases' , 'conv4r_biases', 'conv4r_weights']
        
        # variables are grouped into named scope
        with tf.name_scope("input_target_placeholders"): 
            #self.sess.run(tf.compat.v1.global_variables_initializer())
            self.train_data_node = tf.compat.v1.placeholder(tf.float32,shape=(None, self.dim1, self.dim2, self.num_channels))
            self.train_labels_node = tf.compat.v1.placeholder(tf.float32,shape=(None, self.targetDim1,self.targetDim2))
        
        ################################################################################
        # name of weights as specified in config.yml

        # the fist set of weights are as per specification in config 
        with tf.name_scope("weight_Bias_learning_parameters"):
            self.conv1_weights = tf.Variable(
                tf.random.truncated_normal([self.conf['conv1'][0],self.conf['conv1'][1], self.num_channels, self.conf['conv1'][2]],  # 5x5 filter, depth 32.
                                    stddev=0.1,
                                    seed=self.seed))
            self.conv1_biases = tf.Variable(tf.zeros([self.conf['conv1'][2]]))

            self.conv2_weights = tf.Variable(
                tf.random.truncated_normal([self.conf['conv2'][0],self.conf['conv2'][1], self.conf['conv1'][2], self.conf['conv2'][2]],
                                    stddev=0.1,
                                    seed=self.seed))
            self.conv2_biases = tf.Variable(tf.constant(0.1, shape=[self.conf['conv2'][2]]))
            
            self.conv3_weights = tf.Variable(
                tf.random.truncated_normal([self.conf['conv3'][0],self.conf['conv3'][1], self.conf['conv2'][2], self.conf['conv3'][2]],  # 5x5 filter, depth 32.
                                    stddev=0.1,
                                    seed=self.seed))

            self.conv3_biases = tf.Variable(tf.constant(0.1, shape=[self.conf['conv3'][2]]))

            self.conv4_weights = tf.Variable(
                tf.random.truncated_normal([self.conf['conv4'][0],self.conf['conv4'][1], self.conf['conv3'][2], self.conf['conv4'][2]],  # 5x5 filter, depth 32.
                                    stddev=0.1,
                                    seed=self.seed))
            self.conv4_biases = tf.Variable(tf.constant(0.1, shape=[self.conf['conv4'][2]]))
            
        ################################################################################

        ################################################################################
        # these weights help adjust final shape of output
            self.conv4r = tf.Variable(
                tf.random.truncated_normal([1,2, 2, 1],  
                                    stddev=0.1,
                                    seed=self.seed))
            # this is the weight for the transpose 2d used for bilinear uppsampling
            self.weights = tf.Variable(
                tf.random.truncated_normal([4,4, 16, 16],  
                                    stddev=0.1,
                                    seed=self.seed))

            self.conv4r_biases = tf.Variable(tf.constant(0.1, shape=[1]))
        ################################################################################

        return self.conv4_biases



    def __loss(self ,y, eps=1e-12,cost_name='diceCoeff'):
        '''

        input :  
                y : the predicted segmentation
                cost_name: name of the cost function to be used. check the config file
        output :
                loss: the loss as computed against the true segmentation

        '''

        ################################################################################
        # compute intersection
        A_intersect_B = tf.reduce_sum(y * self.train_labels_node, axis=[0, 1, 2])
        # cimpute union
        A_plus_B =  tf.reduce_sum(y, axis=[0, 1, 2]) + tf.reduce_sum(y, axis=[0, 1, 2])
        ################################################################################

        ################################################################################
        # determine the cost function to be used
        if cost_name == self.conf['loss2']:
            # deniminator for dicecoeff
            denominator = A_plus_B
        elif cost_name == self.conf['loss1']:
           # denominator for intersection of union
           A_union_B = A_plus_B - A_intersect_B
           denominator = A_union_B
        ################################################################################   
        # one can use this loss
        loss = abs(tf.reduce_sum(-(2 * A_intersect_B / (eps + denominator))))

        ################################################################################
        # the loss function computation is also simplified by me below
        # feel free to skip this, It is work in progress
        # one can change the final variable name from losses to loss
        # in order to use it as loss function
        shape = self.train_labels_node.get_shape().as_list()        # a list: [None, 9, 2]
        y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])
        dim = np.prod(shape[1:])            # dim = prod(9,2) = 18
        x2 = tf.reshape(self.train_labels_node, [-1, dim])           # -1 means "all"
        x2r = tf.reduce_sum(x2,axis = [1])
        qr = tf.reduce_sum(y,axis=[1])
        sumr =  x2r+qr
        q = tf.multiply(x2,y)
        #r1 = tf.reduce_sum(q,axis=[0]) 
        r1 = tf.reduce_sum(q,axis=[1])
        r2 = tf.math.divide_no_nan(r1,sumr)
        losses= 1-tf.reduce_mean(r2)
        ################################################################################
        return loss

    def __upsampling(self,input, ops='UpsampleNearNeigh'):

      '''
        input : 
                input : ourput from previous layer of architecture (conv3)
                ops: the kind of upsampling to perform. Check with the config file
        output:  upsampled input


      '''

      ################################################################################
      # check whcih upsample technique is chosen
      # return output accordingly
      if self.upSampleOps ==self.conf['Upsample2']:
          k = input.get_shape().as_list()
          output = tf.image.resize_nearest_neighbor(input, (2*k[1],2* k[2]),align_corners=True)
          return  output
      elif self.upSampleOps  == self.conf['Upsample1']:   
          return self.__upsample_layer(input) # this function is for tiled bilinear upsampling
      ################################################################################

    
    def __upsample_layer(self,input,
                       n_channels=16, name='deconv', upscale_factor=2):

        '''
        bilinear upsampling using conv2d_transpose
        input:
                input: the output from the previous convolution layer
                n_channels: output channel
                upscale facor: the upscaling facot to use
                name : name for the operation
        output: an upsampled version of the input



        '''

        ################################################################################
        # compute kernel size
        kernel_size = 2*upscale_factor - upscale_factor%2 
        # compute scaling factor
        stride = upscale_factor
        strides = [1, stride, stride, 1]
        # get the shape of the input as lisy=t
        k =  input.get_shape().as_list()
        # new heights and wifth shape of the image 
        h = ((k[1] - 1) * stride) + 2*upscale_factor 
        w = ((k[2] - 1) * stride) +  2*upscale_factor 
        # filter shape
        filter_shape = [kernel_size, kernel_size, k[3], k[3]]
        ################################################################################

        ################################################################################
        # output size size of the upsampled input
        new_shape = [self.train_batch_size, h, w, n_channels]
        
        #note that self.weights has been used here
        deconv = tf.nn.conv2d_transpose(input, self.weights, new_shape ,
                                        strides=strides, padding='VALID')
        ################################################################################

        return deconv

    def __predict(self):


        """The Model definition.

            input : Nil
            output: the result of the convolution of the layers of the network 
                    when tensorflow launches a computational graph


        """


        # 2D convolution, with 'SAME' padding (i.e. the output feature map has
        # the same size as the input). Note that {strides} is a 4D array whose
        # shape matchess the data layout: [image index, y, x, depth].

        ################################################################################
        # variable scope and name . note use of reuse. this is important for 
        # the sampling
        with tf.variable_scope("weight_Bias_Convolution_rate", reuse=tf.AUTO_REUSE):
            
            

            #paddings = tf.constant([[0,0],self.conf['padding1'],self.conf['padding1'],[0,0]])

            #conv1pad = tf.pad(self.train_data_node, paddings, "CONSTANT")
            #tf.print(self.train_data_node)
            ################################################################################
            # first convolution layer note the padding
            paddings = [[0,0],self.conf['padding1'],self.conf['padding1'],[0,0]]
            conv1 = tf.nn.conv2d(self.train_data_node,self.conv1_weights,
                                strides=[1, 1, 1, 1],padding=paddings)

            

            # Bias and rectified linear non-linearity.
            relu1 = tf.nn.relu(tf.nn.bias_add(conv1, self.conv1_biases))
            ################################################################################
            
            ################################################################################
            # second convolution layer note padding too
            paddings = tf.constant([[0,0],self.conf['padding2'],self.conf['padding2'],[0,0]])

            conv2Inputpad = tf.pad(relu1, paddings, "CONSTANT")


            conv2 = tf.nn.conv2d(conv2Inputpad,
                                self.conv2_weights,
                                strides=[1, 1, 1, 1],padding='SAME') #her
            relu2 = tf.nn.relu(tf.nn.bias_add(conv2, self.conv2_biases))
            ################################################################################
            

            ################################################################################
            # Max pooling. The kernel size spec {ksize} also follows the layout of
            # the data. Here we have a pooling window of 2, and a stride of 2.
            # MAX POOLING AND THIRD LAYER 
            pool2 = tf.nn.max_pool2d(relu2,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],padding='SAME')


            paddings =tf.constant([[0,0],self.conf['padding3'],self.conf['padding3'],[0,0]])

            conv3Inputpad = tf.pad(pool2, paddings, "CONSTANT")


            

            conv3 = tf.nn.conv2d(conv3Inputpad,
                                 self.conv3_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID')

            relu3 = tf.nn.relu(tf.nn.bias_add(conv3, self.conv3_biases))
            ################################################################################
           
            ################################################################################
            # fourth layer padding
            paddings = tf.constant([[0,0],self.conf['padding4'],self.conf['padding4'],[0,0]])

            # the if statement select the type of upsampling
            # the first upsampling uses bilinear trans2dconv
            if self.upSampleOps == self.conf['Upsample1']:
            
                up =  self.__upsampling(relu3)

                #up = self.upsample_layer(relu3)
                last  = tf.nn.conv2d(up,
                                 self.conv4_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') # here

                lasts = tf.nn.conv2d(last,
                                 self.conv4r,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME') # here
                
                out = tf.squeeze(lasts, [3])

                return out 
            ################################################################################
            ################################################################################
            else:
            ### upsampling Heree 
            # the secinf upsampling uses resize with nearest neighbor
                
                upsample = self.__upsampling(relu3)
                conv4Inputpad = tf.pad(upsample, paddings, "CONSTANT")
                conv4 = tf.nn.conv2d(conv4Inputpad,
                                 self.conv4_weights,
                                 strides=[1, 1, 1, 1],
                                 padding='VALID') # here

                out = tf.nn.bias_add(conv4, self.conv4_biases)

                ##  THIS LAYER IS ADDED TO CONTROL SHAPE 
                last = tf.nn.conv2d(out,
                                 self.conv4r,
                                 strides=[1, 1, 1, 1],
                                 padding='SAME') # here
                # this layer is added for final resizing
                output1 = tf.compat.v1.image.resize_nearest_neighbor(last, (self.targetDim1,self.targetDim2),align_corners=True)
                last = tf.squeeze(output1, [3])
                return last
            ################################################################################
    


    def runModel(self,data,labels, train=True,lr =1e-6 , batch=3):
        '''
        input: 
                data : train data
                labels: train labels
                train: boolean , check if to train
                lr : learning rate
                batch: batch size
        output:
                loss: the dicecoff loss
                predict: predicted image segment

        '''

        ################################################################################
        # set training to true or false and initialise leanring rate
        self.train = train
        self.lr = lr

        ################################################################################
        
        ################################################################################
        self.train_batch_size  = batch # set batch size
        ################################################################################

        ################################################################################
        # this is where we evaluate the loss and get the segment prediction
        with tf.name_scope("segmentation_output"):
            predict = self.sess.run(self.__predict(),feed_dict={self.train_data_node:data})
            self.y_ = predict
            
            loss = self.sess.run(self.__loss(self.y_ ),feed_dict={self.train_labels_node:labels})
            #predict = self.__predict()
            self.loss = loss
        ################################################################################

        ################################################################################
        # add regularization here
        with tf.name_scope("regularised_error"):
            loss = self.sess.run(self.__loss(self.y_ ),feed_dict={self.train_labels_node:labels})
            regularizers = (tf.nn.l2_loss(self.conv1_weights) + tf.nn.l2_loss(self.conv2_weights) +
                        tf.nn.l2_loss(self.conv3_weights) + tf.nn.l2_loss(self.conv4_weights))
            # Add the regularization term to the loss.
            loss += 5e-4 * regularizers
        ################################################################################  

        ################################################################################
        # if we are to train the model then run the parts below
        # which consist of optimization and optimizer
        if self.train:   
            # compute gradients
            # here I intended to normalize gradient but this is left out for now
            #all_grads_node = tf.gradients(loss, self.all_params_names )
            
            #all_grad_norms_node = []
        ################################################################################

        ################################################################################
        # tensorboard summary below
            # Create a summary to monitor MSE
            losses = tf.compat.v1.summary.scalar("errors_Summary",loss)
            summaryHistogram = []
            weight_stats1=tf.compat.v1.summary.histogram ("weights4_Histogram",self.conv4_weights)
            weight_stats2=tf.compat.v1.summary.histogram ("weights3_Histogram",self.conv3_weights)
            weight_stats3=tf.compat.v1.summary.histogram ("weights4r_Histogram",self.conv4r)
            weight_stats4=tf.compat.v1.summary.histogram ("weights2_Histogram",self.conv2_weights)
            weight_stats5=tf.compat.v1.summary.histogram ("weights4_Histogram",self.conv1_weights)

        ################################################################################
        
        ################################################################################
            # learning rate
            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.compat.v1.train.exponential_decay(self.lr,
            global_step,
            self.targetDim1 , #decay step
            0.99, # decay rate
            staircase=False)

            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
        ################################################################################

        ################################################################################
            # merged the summary statistics for model

            merged_summary_op = tf.compat.v1.summary.merge ([weight_stats1,weight_stats2,weight_stats3,weight_stats4
                                   ,weight_stats5,losses])
        ################################################################################

        ################################################################################
            # Use Adam optimizer for the optimization.
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate, 0.01).minimize(loss, global_step=global_step)
            #optimizer = tf.train.AdadeltaOptimizer(learning_rate, 0.01).minimize(loss, global_step=batch)
        ################################################################################

        ################################################################################    
            # uncomment to write, this consumes a lot of memory
            ################################################################################
            #self.writer.add_summary(self.sess.run((merged_summary_op)))
            ################################################################################
            self.saver.save(self.sess, self.savepath  + "model.ckpt")
        ################################################################################
            
            
        # return loss and predicted segment   
        return   tf.convert_to_tensor(self.loss) , tf.convert_to_tensor(self.y_)





