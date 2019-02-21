import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import scipy.io as sio
from tqdm import tqdm
import numpy as np
import time
import os

class AI_DEMO(object):
    def __init__(self,
                 sess,
                 image_size_FE,
                 image_size_PE,
                 batch_size,
                 c_dim,
                 test_FE,
                 test_PE):
        """

        :param sess: open a session for training
        :param image_size_FE: Size of training data in x direction (after data augmentation: cropping)
        :param image_size_PE: Size of training data in y direction
        :param batch_size: Number of training samples every iteration
        :param c_dim: To determine the fourth channel of Data, used for multi-contrast image or complex image;(for example, data_size: batch_size*image_size_FE*image_PE*c_dim)
        :param test_FE: Size of testing data in x direction (without augmention)
        :param test_PE: Size of testing data in y direction (without augmention)
        """
        self.sess = sess
        self.image_size_FE = image_size_FE
        self.image_size_PE = image_size_PE
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.test_FE = test_FE
        self.test_PE = test_PE

    def tfrecord_read_dataset(self, batch_size, size_FE, size_PE, Filenames, c_dim, training):
        """
        read binary data from filenames
        Argument:
            batch_size:
            size_FE: FE Num of data
            size_PE: PE Num of data
            Filenames: name of tfrecord
            c_dim: channel
            training: whether shuffle or not
        return: batch_x, batch_y
        """

        def parser(record):
            features = tf.parse_single_example(record,
                                               features={
                                                   'low_CompI': tf.FixedLenFeature([], tf.string),
                                                   'CompI': tf.FixedLenFeature([], tf.string)
                                               })
            low = tf.decode_raw(features['low_CompI'], tf.float32)
            low = tf.reshape(low, [crop_patch_FE, crop_patch_PE, Num_CHANNELS])

            high = tf.decode_raw(features['CompI'], tf.float32)
            high = tf.reshape(high, [crop_patch_FE, crop_patch_PE, Num_CHANNELS])
            return low, high

        crop_patch_FE = size_FE
        crop_patch_PE = size_PE
        Num_CHANNELS = c_dim
        batch_size = batch_size
        if training == True:
            buffer_size = 20000
        else:
            buffer_size = 1
        # output file name string to a queue
        dataset = tf.data.TFRecordDataset(Filenames)
        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size)
        itertor = dataset.make_one_shot_iterator()
        final = itertor.get_next()
        return final

    def loss_SSIM(self, y_true, y_pred):
        """
        define loss function between Ground_truth and pred of network
        :param y_pred:
        :return: losss
        """
        ssim = tf.image.ssim(y_true, y_pred, max_val=1)
        return tf.reduce_mean((1.0 - ssim) / 2, name='ssim_loss')

    def model(self, images, is_train = False, reuse=False):
        """
        The network including three convolution layers
        :param images: input of network
        :param is_train:  To determine the parameters of Batch Normalization, no use here
        :param reuse: if true, reuse the name of network layers
        :return:
        """
        # ============= To initialize the parameters of network layers ==========#
        # ============  w_int, b_init: initial value of weight and bias of convontion kernels in network====== #
        w_init = tf.truncated_normal_initializer(stddev=0.01)# stddev can be changed for different problems
        b_init = tf.constant_initializer(value=0.0) # value usually 0

        with tf.variable_scope('srcnn', reuse=reuse): # define scope names of tensors, making thing convenient in monitoring tools, like tensorboard
            tl.layers.set_name_reuse(reuse) # reuse name when try to validate during training, True: validation
            inputs = tl.layers.InputLayer(images, name='inputs')
            conv1 = tl.layers.Conv2d(inputs, 64, (9, 9), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='conv1') # convolution layers:   kernel size 9*9, output_channel 64; activation function: relu
            conv2 = tl.layers.Conv2d(conv1, 32, (3, 3), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='conv2') # convolution layers:   kernel size 3*3, output_channel 32; activation function: relu
            conv3 = tl.layers.Conv2d(conv2, self.c_dim, (5, 5), act=None, padding='SAME', W_init=w_init, b_init=b_init,
                                     name='conv3') # convolution layers:   kernel size 5*5, output_channel 1; activation function: None

            return conv3.outputs

    def build_model(self,config):
        """
        To define the placeholder and loss function during the whole training
        1) To define the placeholder for input and output of neural network;
        2) To define loss functions for training and testing, validating
        :return:
        """
        #======================== define placeholer =================================#
        self.images = tf.placeholder(tf.float32, [None, self.image_size_FE, self.image_size_PE, self.c_dim], name='images')
        self.labels = tf.placeholder(tf.float32, [None, self.image_size_FE, self.image_size_PE, self.c_dim], name='labels')
        self.validation_images = tf.placeholder(tf.float32, [None, self.test_FE, self.test_PE, self.c_dim], name='validation_images')
        self.validation_labels = tf.placeholder(tf.float32, [None, self.test_FE, self.test_PE, self.c_dim], name='validation_labels')

        #========================= tensor: output of the network ====================#
        self.pred = self.model(self.images, is_train = True, reuse = False) # training
        self.validation_pred = self.model(self.validation_images, is_train = False, reuse= True) # validation

        #========================== Loss function caclucation =============== #
        self.preding_loss = self.loss_SSIM(self.pred, self.labels)# Loss between output of net with ground_truth for training
        self.srcing_loss = self.loss_SSIM(self.images, self.labels)# Loss between input of net with ground_truth for training
        self.validation_preding_loss = self.loss_SSIM(self.validation_pred, self.validation_labels) # Loss between output of net with ground_truth for validation
        self.validation_srcing_loss = self.loss_SSIM(self.validation_images, self.validation_labels) # Loss between input of net with ground_truth for validation

        #========================= define the option during training================#
        self.saver = tf.train.Saver() # save model option

        #============ define the optimization of neural network =============#
        self.train_op = tf.train.AdamOptimizer(config.LEARNING_RATE).minimize(self.preding_loss)

    def train(self, config):
        """
       Load data and begin train
        :param config:
        :return:
        """
        #============================== Load Data ==================================================#
        train_data = self.tfrecord_read_dataset(self.batch_size,config.image_size_FE,config.image_size_FE,config.tfrecord_train,self.c_dim, True)

        #=================================== initlize variables and restore model=================#
        tf.global_variables_initializer().run()
        # self.saver.restore(self.sess, config.restore_model_filename_3channel)

        #====================================== define array of loss values===========================#
        test_src_loss = np.zeros((config.TESTING_NUM, 1))
        test_pred_loss = np.zeros((config.TESTING_NUM, 1))
        train_src_loss = np.zeros((config.TESTING_NUM, 1))
        train_pred_loss = np.zeros((config.TESTING_NUM, 1))
        # =======================================Training Cycle=====================================#
        if True:
            print("Now Start Training...")
            best_mse = 1
            early_stop_number =100
            for epoch in tqdm(range(config.epoch)):
                # =================Run by batch images =======================#
                batch_xs, batch_ys = self.sess.run(train_data) # Get training data for training
                # ============== training =======================#
                ## ==== train_op: can calculate the descent gradient for next epoch
                # ====== preding_loss: printf to see the loss
                # ===== out: output of current net parameters
                _, err, out = self.sess.run([self.train_op, self.preding_loss, self.pred], feed_dict = {self.images: batch_xs, self.labels: batch_ys})

                # ========================= Save model ===============#
                if epoch % 100 == 0:
                    print('epoch %d training_cost => %.7f ' % (epoch, err))
                    save_path = self.saver.save (self.sess, config.save_model_filename)
                if epoch % 1000 == 0:
                    self.saver.restore(self.sess, config.save_model_filename)
        self.sess.close() # close the session

    def pred_test(self, config):
        #======= Get testing data (tensor)  and restore model=============#
        batch_valid = self.tfrecord_read_dataset(1,config.test_FE,config.test_PE,config.tfrecord_test,self.c_dim,False)
        self.saver.restore(self.sess, config.save_model_filename) # restore model parameters saved during training

        #============= Initialize array for caculating MSE value==============#
        test_src_mse = np.zeros((config.TESTING_NUM, 1))
        test_pred_mse = np.zeros((config.TESTING_NUM, 1))
        recon = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')
        high_res_images = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')
        low_res_images = np.zeros(( self.test_FE, self.test_PE, config.TESTING_NUM, self.c_dim)).astype('float32')

        #============== Begin Testing===================#
        for ep in range(config.TESTING_NUM):
            batch_xs_validation, batch_ys_validation = self.sess.run(batch_valid) # Get testing data for every iteration

            recon[:,:,ep,:], high_res_images[:,:,ep,:], low_res_images[:,:,ep,:] = self.sess.run([self.validation_pred, self.validation_labels, self.validation_images],
                                                              feed_dict={self.validation_images: batch_xs_validation,
                                                                         self.validation_labels: batch_ys_validation}) # get network output, ground_truth, input of network

            test_src_mse[ep], test_pred_mse[ep] = self.sess.run(
                [self.validation_srcing_loss, self.validation_preding_loss],
                feed_dict={self.validation_images: batch_xs_validation,
                           self.validation_labels: batch_ys_validation}) # calculate the loss between  network output with ground_truth, input with ground_truth
            print('ave_src_MSE: %.7f,ave_pred_MSE: %.7f' % (test_src_mse[ep], test_pred_mse[ep]))
        print('mean ave_src_MSE: %.7f,mean ave_pred_MSE: %.7f' % (test_src_mse.mean(), test_pred_mse.mean()))


        self.sess.close()