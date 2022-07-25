import sys
from keras.layers import Multiply,AveragePooling2D,ReLU,Lambda,Activation,multiply,Average,add,Dense, Conv2D, Input, concatenate, MaxPool2D, UpSampling2D, Concatenate, Conv2DTranspose ,MaxPooling2D,Dropout
from keras import layers
from keras.optimizers import Adam,Adadelta
from keras.models import Model
from keras.models import load_model
from keras.utils import get_file
from keras.models import model_from_json
from keras.models import Sequential
from keras.models import model_from_json
import keras        
import keras.backend as K
from keras.models import Sequential




import segmentation_models as sm
from segmentation_models.utils import set_trainable
import tensorflow as tf
import os.path


class ResearchModels():    
    def __init__(self,modelname,width=None,height=None,dim=3,verb=0):
        self.modelname=modelname
        self.width=width
        self.height=height
        self.dim=dim

        
        if 'LRDNet' in modelname:
            print("***** Loading LRDNet proposed model*****")
            self.model = self.LRDNet()    
         
                        
        if verb==1:
            self.model.summary()
            print('*******Total parameters********',self.model.count_params())

    def DiceLoss(self,y_true, y_pred):
        smooth=1e-6  
        gama=2        
        y_true, y_pred = tf.cast(
            y_true, dtype=tf.float32), tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + smooth
        denominator = tf.reduce_sum(
            y_pred ** gama) + tf.reduce_sum(y_true ** gama) + smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

    def get_kwargs(self):
            return {
                'backend': keras.backend,
                'layers': keras.layers,
                'models': keras.models,
                'utils': keras.utils,
            }     
    def dice_coef(self,y_true, y_pred):
        smooth = 1e-5
        
        y_true = tf.round(tf.reshape(y_true, [-1]))
        y_pred = tf.round(tf.reshape(y_pred, [-1]))
        
        isct = tf.reduce_sum(y_true * y_pred)
        
        return 2 * isct / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))

    def iou_coef(self,y_true, y_pred):
      smooth=1e-5
      intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
      union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersection
      iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
      return iou

    def iou_loss(self,y_true, y_pred):
        return 1.0 - self.iou_coef(y_true, y_pred)        
      
    
    def dice_coef_loss(self,y_true, y_pred):
        return 1 - self.dice_coef(y_true, y_pred)
        
    def down(self,input_layer, filters, pool=True):
        filters=int(filters)    
        conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(input_layer)
        residual = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
        if pool:
            max_pool = MaxPool2D()(residual)
            #max_pool=GlobalWeightedAveragePooling2D()(residual)
            return max_pool, residual
        else:
            return residual

    def up(self,input_layer, residual, filters):
        filters=int(filters)
        upsample = UpSampling2D(interpolation='bilinear')(input_layer)
        upconv = Conv2D(filters, kernel_size=(2, 2), padding="same")(upsample)
        concat = Concatenate(axis=3)([residual, upconv])
        conv1 = Conv2D(filters, (3, 3), padding='same', activation='relu')(concat)
        conv2 = Conv2D(filters, (3, 3), padding='same', activation='relu')(conv1)
        return conv2   
       
    def TransNet(self, img, ADI,filters=1):
        assert img.shape != ADI.shape, ("Input shape mismatched: Shapes must be same ", img.shape, ADI.shape)
        f=int(ADI.shape[3])
        theta_a = Conv2D(f, [1, 1], strides=[1, 1], padding='same')(ADI)
        theta_b = Conv2D(f, [1, 1], strides=[1, 1], padding='same')(ADI)
        x1=Multiply()([theta_a,ADI])
        x1=add([x1, theta_b])
        x2 = Concatenate(axis=3)([x1, img])         
        return x2

    def fuse(self,a,b,c,d):
        f=int(a.shape[3])
        t_a = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(a)
        t_b = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(b)
        t_c = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(c)
        t_d = Conv2D(f, [3, 3], strides=[1, 1], padding='same')(d)
        
        x1=add([t_a, t_b])
        x1=add([x1, t_c])
        x1=add([x1, t_d])
        return x1

                
    def get_backbone(self,backbone):
    
        #one of : inceptionv3,mobilenetv2,mobilenet,ResNet34, ResNet50, ResNet101,ResNet152, resnext50, resnext101, vgg16 , vgg19,inceptionresnetv2
        #backbone='EfficientNetB6' 
        
        ##########################################################
        if backbone=='EfficientNetB6':
            print('**** EfficientNetB6 backbone ****')        
            backbone = sm.Unet('efficientnetb6', encoder_weights='imagenet')
            layer_names=['block6a_expand_activation', 'block4a_expand_activation','block3a_expand_activation', 'block2a_expand_activation']
        elif backbone=='EfficientNetB5':
            print('**** EfficientNetB5 backbone ****')        
            backbone = sm.Unet('efficientnetb5', encoder_weights='imagenet')
            layer_names=['block6a_expand_activation', 'block4a_expand_activation','block3a_expand_activation', 'block2a_expand_activation']            
        elif backbone=='EfficientNetB7':
            print('**** EfficientNetB7 backbone ****')        
            backbone = sm.Unet('efficientnetb7', encoder_weights='imagenet')
            layer_names=['block6a_expand_activation', 'block4a_expand_activation','block3a_expand_activation', 'block2a_expand_activation']            
        elif backbone=='ResNet34':
            #backbone=load_model('./models/weights/resnet34.hdf5')
            backbone = sm.Unet('resnet34', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0']                        
        elif backbone=='ResNet50':
            backbone = sm.Unet('resnet50', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'] 
        elif backbone=='ResNet101':
            backbone = sm.Unet('resnet101', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'] 
        elif backbone=='ResNet152':
            backbone = sm.Unet('resnet152', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'] 
        elif backbone=='resnext50':
            print('**** resnext50 backbone ****')                
            backbone = sm.Unet('resnext50', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0']              
        elif backbone=='resnext101':
            print('**** resnext101 backbone ****')        
            backbone = sm.Unet('resnext101', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0'] 
        elif backbone=='vgg16':
            backbone = sm.Unet('vgg16', encoder_weights='imagenet')
            layer_names=['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2']   
        elif backbone=='vgg19':
            print('**** VGG 19 backbone ****')
            backbone = sm.Unet('vgg19', encoder_weights='imagenet')
            layer_names=['block5_conv3', 'block4_conv3', 'block3_conv3', 'block2_conv2', 'block1_conv2']  
        elif backbone=='mobilenet':
            print('**** mobilenet backbone ****')
            backbone = sm.Unet('mobilenet', encoder_weights='imagenet')
            layer_names=['conv_pw_11_relu', 'conv_pw_5_relu', 'conv_pw_3_relu', 'conv_pw_1_relu'] 
        elif backbone=='mobilenetv2':
            print('**** mobilenetv2 backbone ****')
            backbone = sm.Unet('mobilenetv2', encoder_weights='imagenet')
            layer_names=['block_13_expand_relu', 'block_6_expand_relu', 'block_3_expand_relu','block_1_expand_relu']
        elif backbone=='seresnet18':
            print('**** seresnet18 backbone ****')
            backbone = sm.Unet('seresnet18', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0']   
        elif backbone=='seresnet34':
            print('**** seresnet34 backbone ****')
            backbone = sm.Unet('seresnet34', encoder_weights='imagenet')
            layer_names=['stage4_unit1_relu1', 'stage3_unit1_relu1', 'stage2_unit1_relu1', 'relu0']             
        elif backbone=='inceptionresnetv2':
            print('**** Inceptionresnetv2 backbone ****')
            layer_num=[594, 260, 16, 9]
            backbone = sm.Unet('inceptionresnetv2', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1]
        elif backbone=='inceptionv3':
            print('**** inceptionv3 backbone ****')
            layer_num=[228, 86, 16, 9]
            backbone = sm.Unet('inceptionv3', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1]
        elif backbone=='seresnet50':
            print('**** seresnet50 backbone ****')
            layer_num=[246, 136, 62, 4]
            backbone = sm.Unet('seresnet50', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1]  
        elif backbone=='seresnet101':
            print('**** seresnet101 backbone ****')
            layer_num=[552, 136, 62, 4]
            backbone = sm.Unet('seresnet101', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1] 
        elif backbone=='seresnet152':
            print('**** seresnet152 backbone ****')
            layer_num=[858, 208, 62, 4]
            backbone = sm.Unet('seresnet152', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1]             
        
        elif backbone=='seresnext50':
            print('**** seresnext50 backbone ****')
            layer_num=[1078, 584, 254, 4]
            backbone = sm.Unet('seresnext50', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1]         
        elif backbone=='seresnext50':
            print('**** seresnext50 backbone ****')
            layer_num=[1078, 584, 254, 4]
            backbone = sm.Unet('seresnext50', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1]  
        elif backbone=='seresnext101':
            print('**** seresnext101 backbone ****')
            layer_num=[2472, 584, 254, 4]
            backbone = sm.Unet('seresnext101', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1] 
        elif backbone=='senet154':
            print('**** senet154 backbone ****')
            layer_num=[6884, 1625, 454, 12]
            backbone = sm.Unet('senet154', encoder_weights='imagenet')
            layer_names=[]
            #convert layer index to layer names
            for idx, layer in enumerate(backbone.layers):         
                if idx in layer_num:
                    layer_names.append(layer.name) 
            layer_names=layer_names[::-1] 
        return backbone,layer_names  

    def LRDNet(self): # reduced filter sizes to optimize performance
        # Changed last layer to relu
        print('********** LRDNet **********')
        height=self.height
        width=self.width
        backbone,layer_names=self.get_backbone('vgg19')
        
        ### make the pre-trained layer trainable
        backbone.compile('Adam', 'binary_crossentropy', ['binary_accuracy'])
        set_trainable(backbone)

        input_layer = Input(shape = [height, width, 3])
        input_layer_2 = Input(shape = [height, width, 3])
        
        
        # Flavours, try out all of them
        if 'V1' in self.modelname: #<<< -- LRDNet S/L
            l1_flt=8
            col_2_F=64
            col_3_F=32
            col_4_F=16
            last_filters=256
        
        if 'V2' in self.modelname:   ## << works well (visual inspection), but couldnt test on evaluation server, try it yourself.
            l1_flt=16
            col_2_F=16
            col_3_F=8
            col_4_F=4
            last_filters=64
            
        
        if 'V3' in self.modelname: #<<< --  LRDNet +
            l1_flt=64
            col_2_F=64
            col_3_F=32
            col_4_F=16
            last_filters=256


        activation='relu'

        ########## ADI Branch
        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[3]).output)
        x = im(input_layer_2)
        col_1_1_A = Conv2D(l1_flt, (3, 3), padding='same', activation=activation)(x)
        
        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[2]).output)
        x = im(input_layer_2) 
        col_1_2_A = Conv2D(l1_flt*2, (3, 3), padding='same', activation=activation)(x)
        
        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[1]).output)
        x = im(input_layer_2) 
        col_1_3_A = Conv2D(l1_flt*4, (3, 3), padding='same', activation=activation)(x)        


        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[0]).output)
        x = im(input_layer_2) 
        col_1_4_A = Conv2D(l1_flt*8, (3, 3), padding='same', activation=activation)(x) 
        
   
        col_2_4_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_4_A)
        col_2_3_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_3_A)
        col_2_2_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_2_A)
        col_2_1_A = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_1_A)
        
        
        

        col_3_1_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_1_A)
        col_3_2_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_2_A)
        col_3_3_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_3_A)
        col_3_4_A = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_4_A) 
        
        
        col_4_1_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_1_A)
        col_4_2_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_2_A)
        col_4_3_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_3_A)
        col_4_4_A = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_4_A)         


        upsample_1_A = UpSampling2D(interpolation='bilinear',size=(2,2))(col_4_1_A)
        upsample_2_A = UpSampling2D(interpolation='bilinear',size=(4,4))(col_4_2_A)
        upsample_3_A = UpSampling2D(interpolation='bilinear',size=(8,8))(col_4_3_A)
        upsample_4_A = UpSampling2D(interpolation='bilinear',size=(16,16))(col_4_4_A)
        
        
        #out_A = Conv2D(filters=last_filters, kernel_size=(3, 3), padding='same',activation="sigmoid")(cat_A)
        
        #### Image branch
        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[3]).output)
        x = im(input_layer)
        col_1_1 = Conv2D(l1_flt, (3, 3), padding='same', activation=activation)(x)
        col_1_1=self.TransNet(col_1_1,col_1_1_A) 
        
        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[2]).output)
        x = im(input_layer) 
        col_1_2 = Conv2D(l1_flt*2, (3, 3), padding='same', activation=activation)(x)
        col_1_2=self.TransNet(col_1_2,col_1_2_A)
        
        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[1]).output)
        x = im(input_layer) 
        col_1_3 = Conv2D(l1_flt*4, (3, 3), padding='same', activation=activation)(x)
        col_1_3=self.TransNet(col_1_3,col_1_3_A)


        im = Model(inputs=backbone.input,outputs=backbone.get_layer(layer_names[0]).output)
        x = im(input_layer) 
        col_1_4 = Conv2D(l1_flt*8, (3, 3), padding='same', activation=activation)(x) 
        col_1_4=self.TransNet(col_1_4,col_1_4_A)
   
        col_2_4 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(col_1_4)
        col_2_4=self.TransNet(col_2_4,col_2_4_A)        
        upsample = UpSampling2D(interpolation='bilinear')(col_2_4)
        x=self.TransNet(col_1_3,upsample)        
        col_2_3 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        col_2_3=self.TransNet(col_2_3,col_2_3_A)        
        
        
        upsample = UpSampling2D(interpolation='bilinear')(col_2_3)
        x=self.TransNet(col_1_2,upsample)                
        col_2_2 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        col_2_2=self.TransNet(col_2_2,col_2_2_A)        
        
        
        upsample = UpSampling2D(interpolation='bilinear')(col_2_2)
        x=self.TransNet(col_1_1,upsample)          
        col_2_1 = Conv2D(col_2_F, (3, 3), padding='same', activation=activation)(x)
        col_2_1=self.TransNet(col_2_1,col_2_1_A)        
        
        col_3_1 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_1)
        col_3_1=self.TransNet(col_3_1,col_3_1_A)        
        
        
        col_3_2 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_2)
        col_3_2=self.TransNet(col_3_2,col_3_2_A)               
        
        col_3_3 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_3)
        col_3_3=self.TransNet(col_3_3,col_3_3_A)                
        
        col_3_4 = Conv2D(col_3_F, (3, 3), padding='same', activation=activation)(col_2_4)
        col_3_4=self.TransNet(col_3_4,col_3_4_A)               
        
        
        col_4_1 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_1)
        col_4_2 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_2)
        col_4_3 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_3)
        col_4_4 = Conv2D(col_4_F, (3, 3), padding='same', activation=activation)(col_3_4)         


        upsample_1 = UpSampling2D(interpolation='bilinear',size=(2,2))(col_4_1)
        upsample_2 = UpSampling2D(interpolation='bilinear',size=(4,4))(col_4_2)
        upsample_3 = UpSampling2D(interpolation='bilinear',size=(8,8))(col_4_3)
        upsample_4 = UpSampling2D(interpolation='bilinear',size=(16,16))(col_4_4)
        
        
        cat=self.fuse(upsample_1,upsample_2,upsample_3,upsample_4)
        cat_A=self.fuse(upsample_1_A,upsample_2_A,upsample_3_A,upsample_4_A)

        cat=Concatenate(axis=3)([cat, cat_A])
        
        out = Conv2D(filters=last_filters, kernel_size=(3, 3),padding='same', activation="sigmoid")(cat)        
        
        out = Conv2D(filters=1, kernel_size=(1, 1), padding='same',activation="sigmoid")(out)
        model = Model(inputs=[input_layer,input_layer_2], outputs=[out])
        
        #Defining Optimizer and loss settings
        optimizer=Adam(5e-6)
        metrics=[self.iou_coef]
        model.compile(optimizer=optimizer, loss=self.iou_loss, metrics=metrics)
        return model