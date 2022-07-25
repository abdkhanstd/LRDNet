import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
################################### Clean Warning #############
import numpy as np
np.seterr(all="ignore")
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
def warn(*args, **kwargs):
    pass
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_AFFINITY"] = "noverbose"

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(3)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

    

################################### Clean Warning #############
from numpy.random import seed
import tensorflow
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import os
import sys
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint,ReduceLROnPlateau
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger

from keras.models import load_model
from keras.optimizers import Adam
    

import keras.backend as K



# Our internal functions and libraries
from models.models import ResearchModels   # For models
from utils.data import DataSet  # For loading datasets
from utils.data_aug import DataSet_aug  # For loading datasets

from utils.generator import train_generator, val_generator

os.system('clear')


def dice_coef(y_true, y_pred):
    smooth = 1e-5
    
    y_true = tf.round(tf.reshape(y_true, [-1]))
    y_pred = tf.round(tf.reshape(y_pred, [-1]))
    
    isct = tf.reduce_sum(y_true * y_pred)
    
    return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def iou_coef(y_true, y_pred):
  smooth=1
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou    

def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)     
# Model List: LRDNet, LUNet, CapsNetR1, CapsNetR3 , CapsNetBasic, Unet
################## Experiment Settings ##################

model_path='results/LRDNet_LAST_ADI2_MAY25_Weights/LRDNet_LAST_ADI2_MAY25_Weights_Weights/LRDNet_LAST_ADI2_MAY25_Weights.(029)-[0.9602]-[0.9863].hdf5' #


print(model_path)

#tmp=model_path.split('/')
#model=tmp[1]+'_C'+'_ADD'
#model='LRDNet_BB_DI2_ADD'


# Sample model nam, must inclue all relevan keywords

model='LRDNet_LAST_ADI2_MAY25_Weights'

augmentation=False
save_best_only=True
seeding=False
batch_size = 1
patience=15
epochs=1500
save_models=True


    
if seeding:
    seedi=100
    seed(seedi)
    tensorflow.random.set_seed(seedi)



width = 1280
height = 384    

if 'SM' in model_path:
    width = 256
    height = 256
    print('************** Using Size 256 x 256 **************')       
    
if augmentation:    
    train_images=DataSet_aug(model=model,target='train',batch_size=batch_size,width=width,height=height)
    val_images=DataSet_aug(model=model,target='valid',batch_size=batch_size,width=width,height=height)
    aug='[AUGBIG]'
else:
    train_images=DataSet(model=model,target='train',batch_size=batch_size,width=width,height=height)
    val_images=DataSet(model=model,target='valid',batch_size=batch_size,width=width,height=height)
    aug=''


steps_per_epoch = train_images.steps_per_epoch
validation_steps = val_images.validation_steps

train_data=train_images.td
val_data=val_images.vd

# Helper: Save the model
checkpoints_dir='results/'+model
model_dir=checkpoints_dir+'/'+model+'_Weights'

if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)    

checkpointer = ModelCheckpoint(filepath=os.path.join(model_dir, model +aug+'.({epoch:03d})-[{iou_coef:.4f}]-[{val_iou_coef:.4f}].hdf5'),verbose=1,save_best_only=save_best_only)

# Helper: TensorBoard
tb = TensorBoard(log_dir=os.path.join(checkpoints_dir, model+'_logs', model))
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Helper: Save results.
timestamp = time.time()
csv_logger = CSVLogger(os.path.join(checkpoints_dir, model+'_logs', model + str(timestamp) + '.log'))  

# helper early stopper
early_stopper=EarlyStopping(monitor='val_iou_coef', patience=patience, verbose=1)
#early_stopper=EarlyStopping(monitor='iou_coef',mode='max', patience=patience, verbose=1)
# Callbacks
if save_models:  
    #callbacks=[tb, early_stopper, csv_logger,checkpointer]
    callbacks=[tb, csv_logger,checkpointer]
    #callbacks=[tb, early_stopper, csv_logger,checkpointer,reduce_lr]
else:
    callbacks=[tb, early_stopper, csv_logger,reduce_lr]


model=load_model(model_path,compile=False,custom_objects={"iou_coef": iou_coef,"iou_loss":iou_loss})
#Defining Optimizer and loss settings
optimizer=Adam(5e-6)
metrics=[iou_coef]
model.compile(optimizer=optimizer, loss=iou_loss, metrics=metrics) 

history=model.fit_generator(train_data, steps_per_epoch, epochs=epochs, verbose=1,
                        callbacks=callbacks, validation_data=val_data,
                        validation_steps=validation_steps)

'''    
# Loading the model
rm= ResearchModels(modelname=model,height=height,width=width)
# Training the Network
history=rm.model.fit_generator(train_data, steps_per_epoch, epochs=epochs, verbose=1,
                        callbacks=callbacks, validation_data=val_data,
                        validation_steps=validation_steps)
   '''                     
                        
'''                        
# Plot training & validation iou_score values
plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('Model iou_score')
plt.ylabel('iou_score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
#plt.show()
'''
