# -*- coding: utf-8 -*-

import Generator 
import Discriminator
from config import * 
import tensorflow as tf 
from tensorflow.keras.models import Sequential , Model
from tensorflow.keras.layers import Input
from data import *
import skimage
import datetime
import numpy as np 
from matplotlib import pyplot as plt 
#%%
tf.keras.backend.clear_session()
#%%
def build_VGG(): 
    input_layer = Input(shape= high_resolution_image_shape)
    vgg_model = tf.keras.applications.VGG19( weights='imagenet' , include_top= False )
    vgg_model.outputs = [vgg_model.layers[vgg_out_layer].output]
    output_layer = vgg_model(input_layer)
    model = Model(input_layer , output_layer , name='VGG')
    model.trainable = False 
    return model 

vgg = build_VGG() # build vgg model
vgg.compile(loss= vgg_loss , optimizer=optimizer) # compile vgg



generator = Generator.build_generator() 
discriminator = Discriminator.build_discriminator()
discriminator.compile( loss=disc_loss , optimizer=optimizer )

disc_output_shape = discriminator.output.shape
n , h , w , c = disc_output_shape
disc_output_shape = (h , w , c)


lr_input = Input(low_resolution_image_shape , name='lr_input')
hr_input = Input(high_resolution_image_shape, name='hr_input')

fake_hr_images = generator(lr_input)
vgg_features   = vgg(fake_hr_images )
discriminator.trainable=False
validity = discriminator(fake_hr_images )

total_model = Model( [lr_input , hr_input]  , [validity , vgg_features] , name='Total_Model')
total_model.compile( loss= [disc_loss , vgg_loss] , loss_weights=[1e-3 , 1] , optimizer=optimizer)



def sample_images( epoch):
        os.makedirs('images/%s' % dataset_name, exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = load_data(batch_size=2)
        fake_hr = generator.predict(imgs_lr)

        

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%s/%d.png" % (dataset_name, epoch))
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%s/%d_lowres%d.png' % (dataset_name, epoch, i))
            plt.close()
            
            
            
            
            
def train(batch_size= batch_size  , epochs= 500 , sample_interval=50) :
        start_time = datetime.datetime.now()

        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = generator.predict(imgs_lr)

            valid = np.ones((batch_size,)  + disc_output_shape)
            fake  = np.zeros((batch_size,) + disc_output_shape)

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + disc_output_shape)

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = vgg.predict(imgs_hr)

            # Train the generators
            g_loss = total_model.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print ("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                sample_images(epoch)
                
                fake_hr = generator.predict(imgs_hr)
                psnr = tf.image.psnr(imgs_hr , fake_hr , 1.0 )
                psnr = np.average(psnr)
                ssim = skimage.measure.compare_ssim(imgs_hr , fake_hr)
                ssim = np.average(ssim)
                print("psnr = {} , ssim = {}".format(psnr , ssim) )


#%%
train()
generator.save('models\\generator.h5')
discriminator.save('models\\discriminator.h5')
total_model.save('models\\total_model.h5')














#%%