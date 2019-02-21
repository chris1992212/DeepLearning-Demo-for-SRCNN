import tensorflow as tf
import os
from model import AI_DEMO
from easydict import EasyDict as edict
import json
import tensorlayer as tl

config = edict() #


#=================Common Parameters Set for Training =========================#
config.LEARNING_RATE = 0.0001 # learning rate
config.epoch = 250000 # Total iteration numbers
config.early_stop_number = 1000 # Stop training when loss in valiation data is no longer decline after a 1000 times
config.batch_size = 16

#========================== Size of Data, including training, and testing==================#
config.image_size_FE = 80 # size of training data in x direction
config.image_size_PE = 80 # size of training data in y direction
config.c_dim = 3 # dimension of input data

#============================Testing Set======================#
config.Test_Batch_size = 1
config.TESTING_NUM = 96
config.test_FE = 384
config.test_PE = 288 # size of testing data in y direction

#===========================Files: Data Files, Model Files============================#
config.tfrecord_train = os.path.join('train.tfrecord') # training data file
config.tfrecord_test = os.path.join('testing.tfrecord') # testing data file
config.save_model_file = 'Model'
config.save_model_filename = os.path.join(config.save_model_file,'train_model.ckpt') # save model
tl.files.exists_or_mkdir(config.save_model_file)

def main(_): #?
    with tf.Session() as sess:
        AI_demo = AI_DEMO(sess,
                      image_size_FE = config.image_size_FE,
                       image_size_PE=config.image_size_PE,
                      batch_size = config.batch_size,
                      c_dim = config.c_dim,
                      test_FE = config.test_FE,
                      test_PE= config.test_PE
                         )
        AI_demo.build_model(config)
        # AI_demo.train(config)
        AI_demo.pred_test(config)

if __name__=='__main__':
    tf.app.run()