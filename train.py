from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from data import *
from generator import BatchGenerator,random,GLCM
from loss import my_loss1,dice_coef,IoU,csf_loss
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2' # 只显示 Error
from keras.optimizers import Adam
from my_model import ResNetR3_attention_mutil_scale
from keras.callbacks import ModelCheckpoint,EarlyStopping
learn_rate = 1e-4


x_train,y_train ,affine= get_data('train_data/txt_file/','train_data/train/','train_data/train_mask/')   #160,192,160
x_valid,y_valid ,affine1= get_valid_data('train_data/txt_file/','train_data/valid/','train_data/valid_mask/')
x_train,y_train = random(x_train,y_train)
x_valid,y_valid = random(x_valid,y_valid)
myGene = BatchGenerator(x_train,y_train, 1,1, augment = True)
myvaild_Gene = BatchGenerator(x_valid,y_valid, 1,1, augment = False)

model = ResNetR3_attention_mutil_scale((64,64,64,1), (64,64,64,1),(64,64,64,1),(64,64,64,1),(64,64,64,1))
model.summary()
model.compile(optimizer=Adam(lr=learn_rate), loss={'output8':my_loss1,'output16':my_loss1,'output32':my_loss1,'output64':my_loss1,'output1':my_loss1,'output2':my_loss1,'CSF':csf_loss,'GM':csf_loss,'WM':csf_loss},metrics=[dice_coef])
model_checkpoint = ModelCheckpoint('my_1.hdf5', monitor='loss',verbose=1, save_best_only=True)
model_checkpoint1 = ModelCheckpoint('weights.{epoch:02d}-{loss:.4f}.hdf5',monitor='loss',verbose = 1,save_best_only=True,save_weights_only=True,mode='auto',period=1)
early_stop = EarlyStopping(monitor='loss',patience=10)
model.fit_generator(myGene,steps_per_epoch=7360,epochs=20,validation_data=myvaild_Gene,validation_steps = 960,callbacks=[model_checkpoint,model_checkpoint1,early_stop])

