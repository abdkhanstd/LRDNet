import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
'''
def get_session(gpu_fraction=0.3):
    """Assume that you have 6GB of GPU memory and want to allocate ~2GB"""
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
    return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


KTF.set_session(get_session())
'''
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

################################### Clean Warning #############
import warnings
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

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
import numpy as np
import pandas as pd
import tensorflow
from tensorflow.keras.models import Model
import tensorflow as tf
from keras.callbacks import Callback

from skimage.io import imsave
from skimage.color import rgb2gray

import skimage
from skimage.util import img_as_ubyte

from tqdm import tqdm
import os
from PIL import Image,ImageOps
from tensorflow.keras.models import load_model
import numpy as np
import keras.backend as K
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import time as tim
import cv2 as cv
import time
import keras
from models.models import ResearchModels  


test_predict_dir='seg_results_images'
if not os.path.exists(test_predict_dir):
    os.makedirs(test_predict_dir)

class BatchTimeCallback(Callback):
    def on_train_begin(self, logs=None):
        self.batch_times = []
        
    def on_predict_begin(self, logs=None):
        self.start_time_p=time.time()    

    def on_predict_end(self, logs=None):
        stop_time_p=time.time()
        duration =stop_time_p-self.start_time_p
                

    def on_batch_end(self, batch, logs=None):
        self.batch_times.append(time.time())
    def on_predict_batch_begin(self, batch, logs=None):
        self.start_time=time.time()    
        
    def on_predict_batch_end(self, batch, logs=None):
        stop_time=time.time()
        duration =stop_time-self.start_time
        print('Time taken to predict batch :',duration)         

def get_flops(model):


    #run_meta = tf.RunMetadata()
    run_meta= tf.compat.v1.RunMetadata()
    
    opts = tf.profiler.ProfileOptionBuilder.float_operation()

    # We use the Keras session graph in the call to the profiler.
    flops = tf.profiler.profile(graph=K.get_session().graph,
                                run_meta=run_meta, cmd='op', options=opts)

    return flops.total_float_ops  # Prints the "flops" of the model.
    
def get_flops3(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path,custom_objects={"iou_coef": iou_coef,"iou_loss":iou_loss})

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops    

def get_flops2(model):


    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    #
    # Note: Only Ops with RegisterStatistics('flops') defined have flop stats. It
    # also requires complete shape information. It is common that shape is unknown
    # statically. To complete the shape, provide run-time shape information with
    # tf.RunMetadata to the API (See next example on how to provide RunMetadata).
    return tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
        
def write_to_pngfiles(calib,batch_labels, images, next_start):
    target_dir=test_predict_dir  +'/masks/'
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    for i in range(batch_labels.shape[0]):
        img_array = batch_labels[i,:,:,0]*255        
        img = Image.fromarray((img_array).astype(np.uint8))
        # Resize to original file size (optional, if you want to)
        # Otherwise comment out this line
        img=img.resize((1242,375))

        im_name=test_images[next_start]
        im_name=im_name.replace("_","_road_")
        img_path = target_dir + im_name 

        img.save(img_path)
        next_start=next_start+1
    return next_start
        
def test_prediction(model, images,adi, test_dir, test_dir_ADI, batch_size=1):

    count = 0
    I=images
    total_images = images.shape[0]
    total_steps = int(total_images/batch_size)
   
    batch_features = np.zeros((batch_size, height, width,3), dtype='float32')
    batch_features_adi = np.zeros((batch_size, height, width,3), dtype='float32')
    I=np.zeros((batch_size, 375,1242,3), dtype='float32')# for orignnal size image

    batch_labels = np.zeros((batch_size, height, width, 1), np.float32)
    atime=0
    cnt=0
    first=0

    next_start=0
    for steps in range(total_steps):
        ishift = steps*batch_size
        # ignore first run as it gives the most non-optimal time on my shared GPU environment
        for i in range(batch_size):
            # For image
            img_path = test_dir + images[count]  
            calib=img_path.replace('image_2','calib')
            calib=calib.replace('png','txt')
            original_img = Image.open(img_path)
            original_img = original_img.resize((1242,375))            
            resized_img = original_img.resize((width, height))
                     
            I[i,:,:,:] = original_img
            
            img_array = img_to_array(resized_img)/255.            
            batch_features[i,:,:,:] = img_array
            
            # For ADI
            img_path = test_dir_ADI + adi[count]                    
            original_img = Image.open(img_path)
            
            resized_img = original_img.resize((width, height))
            
            img_array = img_to_array(resized_img)/255.
            
            image1=np.array(img_array)## NL
            img_array=image1-np.mean(image1[image1>0])## NL            
            
            batch_features_adi[i,:,:,:] = img_array
            count = count + 1        
        if first==0:
            first=1        
        else:
            first=2        
            t = tim.time()    
            cnt=cnt+1
        batch_time_callback = BatchTimeCallback()
        batch_labels  = model.predict([batch_features,batch_features_adi], batch_size = batch_size, verbose=0)
      
        if first==2:
             atime = atime + tim.time() - t
        next_start=write_to_pngfiles(calib,batch_labels, images, next_start)        

       
    print("Time taken (Average) for one image is: ", (atime/cnt)/batch_size)
                
    return

def iou_coef(y_true, y_pred):
  smooth=1
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
  union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return iou
def iou_loss(y_true, y_pred):
    return 1.0 - iou_coef(y_true, y_pred)   




# List of test images
test_dir = "data/testing/image_2/"
test_dir_ADI = "data/testing/ADI/"
        
test_images = os.listdir(test_dir)
test_images.sort()
test_images = np.array(test_images)    

test_images_ADI = os.listdir(test_dir_ADI)
test_images_ADI.sort()
test_images_ADI = np.array(test_images_ADI)          
        
####################### Write your parameters here ######################################
batch_size = 1
model_path='results/LRDNet_LAST_ADI2_MAY25_Weights/LRDNet_LAST_ADI2_MAY25_Weights_Weights/LRDNet_LAST_ADI2_MAY25_Weights.(225)-[0.9580]-[0.9859].hdf5' #BEST MaxF


## LRDNet-S= V1
## LRDNet-L= V1
## LRDNet+= V3
## Append Vx accordingly 

model='LRDNet_V3'

## V1=Lite normal images and  LITE SM
## V3=Lite normal images 

#############################################################################
os.system('clear')
print('Testing model :',model_path)


width = 1280
height = 384  

if 'SM' in model_path:
    width = 256
    height = 256
    print('************** Using Size 256 x 256 **************')
    
if 'SP' in model_path:
    test_dir = 'data/testing/image_2_sp/'
    print('************** Tesing using super pixel data **************')

if 'DEPTH' in model_path:
    test_dir_ADI = "data/testing/depth_u16/"
    print('************** Tesing using DEPTH data **************')            

if 'ADI2' in model_path:
    print('************** Using our ADI **************')        
    test_dir_ADI = 'data/testing/ADI_2/'




rm= ResearchModels(modelname=model,height=height,width=width,verb=1)
print('******* Total number of FLOPS ********:',get_flops(rm.model)/10**9)

rm.model.load_weights(model_path)
print(rm.model.summary())
print('*******Total parameters********:',rm.model.count_params())


from net_flops import net_flops



#model=load_model(model_path,custom_objects={"iou_coef": iou_coef,"iou_loss":iou_loss})
print('******* Please wait, testing samples')
net_flops(rm.model,table=True)



# Test Prediction
test_prediction(rm.model, test_images,test_images_ADI, test_dir, test_dir_ADI,batch_size=batch_size)

