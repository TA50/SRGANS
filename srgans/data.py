from config import *
import tensorflow as tf
from glob import glob 
import numpy as np 
import imageio 
import skimage
from matplotlib import pyplot as plt
from models import *
from tensorflow.keras.utils import normalize
import os


def load_data(batch_size=1 ) :
    imgs_pathes = glob( "{}\\*.{}".format( dataset_path , img_encoding_type) ) # get the path of all images
    imgs_pathes = np.random.choice(imgs_pathes , batch_size) # take random batch_size of all images
    imgs_list  = [] 
    for img_path in imgs_pathes:
        
            img = imageio.imread( img_path ,
                               pilmode = img_color_mode )
            
            imgs_list.append(img)
       
      
    hr_imgs = [] 
    lr_imgs = []
    for img in imgs_list : 
        img =  skimage.transform.resize(img , output_shape= high_resolution_image_shape, anti_aliasing =True )
        downscaled_img =  skimage.transform.resize(img , output_shape= low_resolution_image_shape, anti_aliasing =True )
        lr_imgs.append(downscaled_img)
        hr_imgs.append(img)
    
    hr_imgs = np.array(hr_imgs)
    lr_imgs = np.array(lr_imgs)
    
    
    #Normalization 
    #hr_imgs = normalize(hr_imgs) 
    #lr_imgs = normalize(lr_imgs) 


    return hr_imgs, lr_imgs
    
    
    
    
    
