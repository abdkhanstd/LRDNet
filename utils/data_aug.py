import random
from PIL import Image
import numpy as np
import pandas as pd
import imageio
import os.path
import re
from keras.preprocessing.image import array_to_img, img_to_array, load_img, ImageDataGenerator
import albumentations as A
import cv2 as cv



class DataSet_aug():
    def __init__(self,model=None,target='train',batch_size=2,width=None,height=None,dim=3):
        random.seed(100)
        np.random.seed(100)
        
        self.batch_size=batch_size
        self.width=width
        self.height=height
        self.dim=dim
        self.target=target
        self.model=model
        
        validation=20 # validation split percentage
        
        validation=validation/100
        
        # Path setting for dataset
        self.image_data_folder = 'data/data_road_aug/train/images/'
        self.ADI_folder = 'data/data_road_aug/train/ADI/'
        self.mask_data_folder = 'data/data_road_aug/train/masks/'

        self.image_data_folder_val = 'data/data_road_aug/val/images/'
        self.ADI_folder_val = 'data/data_road_aug/val/ADI/'
        self.mask_data_folder_val = 'data/data_road_aug/val/masks/'
        

        # List of images ad their masks for training 
        all_images = os.listdir(self.image_data_folder) #Original images
        all_images.sort()
        all_images = np.array(all_images)
        self.all_images=all_images       


        all_images_val = os.listdir(self.image_data_folder_val) #Original images
        all_images_val.sort()
        all_images_val = np.array(all_images_val)
        self.all_images_val=all_images_val 
        
 

               
        self.train_images=all_images
        self.valid_images=all_images_val
        
        self.train_ADI=all_images
        self.valid_ADI=all_images_val
  
        if self.target=='train':
            self.td=self.batch_generator(self.train_images, self.batch_size,'train')
            self.steps_per_epoch=self.train_images.shape[0]/self.batch_size
            return None
        if self.target=='valid':
            self.vd=self.batch_generator(self.valid_images, self.batch_size,'val')        
            self.validation_steps =self.valid_images.shape[0]/self.batch_size            
            return None

        if self.target=='test':
          
            return None            
        
    def batch_generator(self,images, batch_size,target):
            while True:
                ix = np.random.choice(np.arange(images.shape[0]), batch_size)
                batch_features = []
                batch_features_2 = []
                batch_labels = []

                index = 0
                for i in ix:

                    if target=='train':
                        original_img = Image.open(self.image_data_folder + images[i]) 
                        original_ADI = Image.open(self.ADI_folder + images[i]).convert('RGB') 
                        original_mask = Image.open(self.mask_data_folder + images[i])
                        
                        
                    else:
                        original_img = Image.open(self.image_data_folder_val + images[i])
                        original_ADI = Image.open(self.ADI_folder_val + images[i]).convert('RGB')                   
                        original_mask = Image.open(self.mask_data_folder_val + images[i])
                        
                        
                    resized_img = original_img.resize((self.width, self.height))
                    
                    ###################### Hist Eq Start##############
                     R, G, B = cv.split(np.array(resized_img))
                     output1_R = cv.equalizeHist(R)
                     output1_G = cv.equalizeHist(G)
                     output1_B = cv.equalizeHist(B)
                     resized_img = cv.merge((output1_R, output1_G, output1_B))                    
                    ###################### Hist Eq  END ##############
                    
                    image0 = img_to_array(resized_img)/255.
                    batch_features.append(image0)  
                
                    resized_img = original_ADI.resize((self.width, self.height))
                    tmp=np.zeros((self.height, self.width,3), dtype='float32')
                    image1 = img_to_array(resized_img)/255.
                    
                    image1=np.array(image1)## NL <<<<---------------
                    image1=image1-np.mean(image1[image1>0])## NL <<<<---------------
                    
                    batch_features_2.append(image1)
                    

                    resized_mask = original_mask.resize((self.width, self.height))
                    label = img_to_array(resized_mask)/255.
                    batch_labels.append(label[:,:,2])

                
                    batch_features=np.array(batch_features, dtype='float32')
                    batch_features_2=np.array(batch_features_2, dtype='float32')
                    batch_labels=np.array(batch_labels, dtype='float32')
                    batch_labels=batch_labels[:,:,:,np.newaxis]
                    
                    index = index + 1
    
                    yield [batch_features,batch_features_2], batch_labels
                    