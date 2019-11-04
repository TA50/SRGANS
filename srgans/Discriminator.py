import tensorflow as tf 
from tensorflow.keras.layers import Conv2D , Dense , MaxPooling2D , Dropout , Add , Input , Flatten , BatchNormalization 
from tensorflow.keras.models import Model 

from config import * 




# discriminator parameters 
n_res_layers = 8 
disc_filters = 64
disc_hidden_layers = 8
disc_act = tf.keras.activations.relu
disc_initializer = 'random_uniform' 
disc_optimizer = 'adam'
disc_loss = 'binary_crossentropy'
disc_momentum = 0.8
disc_kernal_size = (3,3)

def d_block(pre_layer , k=3 , n = 64 , s=1) :
    m =  Conv2D(filters=n , kernel_size=k ,padding= 'same', strides=s,
                   kernel_initializer=disc_initializer , bias_initializer=disc_initializer)(pre_layer)
    m = BatchNormalization(momentum=disc_momentum)(m)
    m = disc_act(m)
    
    return m
    

def build_discriminator():
    with tf.name_scope('disc_input'):
        input_layer = Input( shape = high_resolution_image_shape , name = 'disc_input_layer')
        layer = Conv2D(filters=disc_filters , kernel_size=disc_kernal_size ,padding = 'same',
                       kernel_initializer=disc_initializer , bias_initializer=disc_initializer )(input_layer)
        layer = disc_act(layer)
    
    with tf.name_scope('hidden_layers'):
        layer = d_block(layer , n=64  , s=2 )
        layer = d_block(layer , n=128 , s=1 )
        layer = d_block(layer , n=128 , s=2 )
        layer = d_block(layer , n=256 , s=1 )
        layer = d_block(layer , n=256 , s=2 )
        layer = d_block(layer , n=512 , s=1 )
        layer = d_block(layer , n=512 , s=2 )
    
    
  
    layer = Dense(1024 , kernel_initializer=disc_initializer , bias_initializer=disc_initializer , name='dense')(layer)
    layer = disc_act(layer)
    output_layer = Dense(1 , activation='sigmoid' , name='output_layer' , 
                         kernel_initializer=disc_initializer , bias_initializer=disc_initializer  )(layer)
    
    model= Model(input_layer , output_layer  , name = 'Discriminator')            
    return model 

