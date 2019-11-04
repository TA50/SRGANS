import tensorflow as tf 
from tensorflow.keras.layers import Conv2D , Dense , MaxPooling2D , Dropout , Add , Input , Flatten ,  BatchNormalization,UpSampling2D

from tensorflow.keras.models import Model 

from config import * 



# Generator Parameters
gen_kernal_size = 3 
gen_filters = 64 
gen_act = tf.keras.activations.relu
gen_initializer = "random_uniform"
n_res_blocks = 16
gen_momentum = 0.8 

def res_block( pre_layer) :
            m = Conv2D(filters=gen_filters , kernel_size=gen_kernal_size ,padding = 'same',
                       kernel_initializer=gen_initializer , bias_initializer=gen_initializer )(pre_layer)
            m = gen_act(m)
            m = BatchNormalization(momentum=gen_momentum)(m)
            
            
            m = Conv2D(filters=gen_filters , kernel_size=gen_kernal_size ,padding = 'same',
                       kernel_initializer=gen_initializer , bias_initializer=gen_initializer )(m)
            m = gen_act(m)
            m = BatchNormalization(momentum=gen_momentum)(m)
            return Add()([m , pre_layer])

def build_generator(): 
    input_layer = Input( shape= low_resolution_image_shape) 
    layer = Conv2D(filters=gen_filters , kernel_size=gen_kernal_size*3 ,padding = 'same',
                       kernel_initializer=gen_initializer , bias_initializer=gen_initializer , name='input_layer' )(input_layer)
    layer = gen_act(layer )
    
    with tf.name_scope('Res_blocks') :
        a = res_block(layer )
        for i in range(n_res_blocks-1):
            a = res_block(a)
        
    with tf.name_scope('post_res_blocks'): 
        a = Conv2D(filters=gen_filters , kernel_size=gen_kernal_size ,padding = 'same',
                       kernel_initializer=gen_initializer , bias_initializer=gen_initializer  )(a)
        a = BatchNormalization(momentum=gen_momentum)(a)
        a = gen_act(a)
        a = Add()([a , layer])
    
    with tf.name_scope('upSampling'): 
        layer = UpSampling2D(size=scale)(a)
        layer = Conv2D(filters=gen_filters , kernel_size=gen_kernal_size ,padding = 'same',
                       kernel_initializer=gen_initializer , bias_initializer=gen_initializer  )(layer)
        layer = gen_act(layer)
        
        
    output_layer = Conv2D(filters=3 , kernel_size=9 ,padding = 'same',activation='tanh',
                       kernel_initializer=gen_initializer , bias_initializer=gen_initializer , name = 'output_layer'  )(layer) 
    
    model = Model(input_layer , output_layer , name = 'Generator')
    return model
    
    