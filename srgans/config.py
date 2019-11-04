batch_size =1
dataset_path = "A:\\Machine learning & Data Science\\datasets\\cptac-cm-512x512"
dataset_name = "cptac-cm-512x512"
resizing_filter = 'bicubic'
vgg_out_layer = 9
# image info
img_encoding_type = 'jpg'
img_color_mode = 'RGB'

#shaping: 
channels  = 3 
scale = 4
high_resolution_image_height = 512
high_resolution_image_width  = 512
high_resolution_image_shape = (high_resolution_image_height , high_resolution_image_width , channels)
low_resolution_image_height = high_resolution_image_height // scale
low_resolution_image_width  = high_resolution_image_width // scale
low_resolution_image_shape  = (low_resolution_image_height , low_resolution_image_width , channels )



#hyper-parameters
 
optimizer = 'adam'
disc_loss = 'binary_crossentropy'
vgg_loss = 'mse'





















    