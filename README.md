# LRDNet
Please note that at this time (due to a busy schedule and some other reasons), only a brief description is provided. However, one can still access the details and re-generate the results.  
This repository contains the official implementation of LRDNet. LRDNet is a lightweight and new method that efficiently detects free road space. The proposed network is lightweight having only 19.5 M parameters (approximately). To date, the LRDNet has the least parameters and the lowest processing time.  

Please note that I have cleaned the code, but haven't tested the cleaned code. If there's any problem, please let me know. Moreover, the code is implemented in Keras as we further want it to work on embedded devices. Pytorch despite being faster than Keras on CPU and GPU environments,  it seems that Pytorch isn't supported on embedded devices that can achieve up to 300 FPS.  


Please refer to the code, the code is self-descriptive. However, here is an overall description of the files:
```
test.py (Test a trained model. The model names need to be specified in test.py i.e., model_path)
train.py (Code for training. Must specify keywords in the model variable)
trainc.py (Continue train, must specify the weights file that needs to be continued)
```
#### Samples results on the KITTI Road Benchmark dataset     

![Sample results]( https://github.com/abdkhanstd/LRDNet/raw/main/images/qres.jpg)
#### FPS and parameter info     

![Table performance]( https://github.com/abdkhanstd/LRDNet/raw/main/images/table.png)

#### Dependencies  
  ```Requirements: Tensorflow-gpu==1.14.0, Keras==2.2.4, tqdm, pillow, numpy Simply run each code.```  
For FLOPS:  
  ``` Download net_flops.py from  https://github.com/ckyrkou/Keras  and keep it on the root folder (if FLOPS calculation is required```  
For Backbones  
  ``` Install Segmentation Models for backbone using instructions from https://github.com/qubvel/segmentation```  
For Augmentation  
  ``` Install Albumentations using details from https://github.com/albumentations-team/albumentations or https://albumentations.ai/```  

##### How to download dataset(s)? 
Please refer to data provider websites  
[KITTI Road Benchmark]( http://www.cvlibs.net/)  
[City Scapes]( https://www.cityscapes-dataset.com/)  
[R2D]( https://sites.google.com/view/sne-roadseg/dataset)  
```Place the dataset into data>testing and data>training  ```  
![Sample results](https://github.com/abdkhanstd/LRDNet/raw/main/images/folder.png)  
``` For Augmentation data, place the data in data_road_aug>train and data_road_aug>val```  





#### Download pre-trained weights for testing 
We provide many pretrained weight files. The models that were used to evaluate on KITTI evaluation server are represented by their names i.e., LRDNet+, LRDNet(s), and LRDNet(L).

The weight files (HDF5 with learning gradients and abstraction layers preserved), the submitted BEV, and HTML evidence of KITTI submissions can be downloaded from [here (Uploading please wait]( http://www.cvlibs.net/)  


##### Please cite As:
Currently, the paper is submitted. Further details will be added soon after the response from the journal. No prepreint uploaded yet



