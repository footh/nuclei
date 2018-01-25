import tensorflow as tf
import numpy as np
import math

IMG_HEIGHT = 256
IMG_WIDTH = 256

FEATURE_ROOT = 64

TCONV_ROOT = 8
TCONV_OUTPUT_CHANNELS = 2 # One for contours, one for segments

DROPOUT_PROB = 0.5

def bilinear_interp_init(shape, dtype=None, partition_info=None):
    """
        Keras customer initializer for bilinear upsampling
        From: https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py#L245
    """
    width = shape[0]
    height = shape[1]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros((width, height), dtype=dtype.as_numpy_dtype())
    for x in range(width):
        for y in range(height):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x,y] = value
    
    tf.logging.info(bilinear.shape)        
    weights = np.zeros(shape, dtype=dtype.as_numpy_dtype())
    for i in range(shape[2]):
        weights[:,:,i,i] = bilinear
        
    tf.logging.info(weights.shape)
    return weights
    #return tf.convert_to_tensor(weights, dtype=dtype, name='bilinear_interp_init')
    #return tf.constant(weights, shape=shape, dtype=dtype, name='bilinear_interp_init')

def build_graph():
    sess = tf.InteractiveSession()

    img_input = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='img_input')
  
    # Better way to do this is, build out all the convolutions first and save the last 3 in different variables, then run the upsampling
    # on the three different convolution points. 
    
    # 0
    net = tf.keras.layers.Conv2D(FEATURE_ROOT, (3,3), padding='same', activation='relu')(img_input)
    
    # 1
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*2, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)
    
    # 2
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*4, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)

    # 3
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*8, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)
    tf.logging.info(f"net after #3: {net}")
    
    c_tconv1 = tf.keras.layers.Conv2DTranspose(TCONV_OUTPUT_CHANNELS, 
                                              (TCONV_ROOT*2, TCONV_ROOT*2), 
                                              strides=(TCONV_ROOT, TCONV_ROOT),
                                              padding='same',
                                              activation='relu',
                                              kernel_initializer=bilinear_interp_init)(net)
    c_tconv1_output = tf.keras.layers.Conv2D(TCONV_OUTPUT_CHANNELS, (1,1), padding='same', activation='relu')(c_tconv1)

    s_tconv1 = tf.keras.layers.Conv2DTranspose(TCONV_OUTPUT_CHANNELS, 
                                              (TCONV_ROOT*2, TCONV_ROOT*2), 
                                              strides=(TCONV_ROOT, TCONV_ROOT),
                                              padding='same',
                                              activation='relu',
                                              kernel_initializer=bilinear_interp_init)(net)
    s_tconv1_output = tf.keras.layers.Conv2D(TCONV_OUTPUT_CHANNELS, (1,1), padding='same', activation='relu')(s_tconv1)

    # 4
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*8, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.Dropout(rate=DROPOUT_PROB)(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)
    tf.logging.info(f"net after #4: {net}")

    c_tconv2 = tf.keras.layers.Conv2DTranspose(TCONV_OUTPUT_CHANNELS, 
                                              (TCONV_ROOT*4, TCONV_ROOT*4), 
                                              strides=(TCONV_ROOT*2, TCONV_ROOT*2),
                                              padding='same',
                                              activation='relu',
                                              kernel_initializer=bilinear_interp_init)(net)
    c_tconv2_output = tf.keras.layers.Conv2D(TCONV_OUTPUT_CHANNELS, (1,1), padding='same', activation='relu')(c_tconv2)

    s_tconv2 = tf.keras.layers.Conv2DTranspose(TCONV_OUTPUT_CHANNELS, 
                                              (TCONV_ROOT*4, TCONV_ROOT*4), 
                                              strides=(TCONV_ROOT*2, TCONV_ROOT*2),
                                              padding='same',
                                              activation='relu',
                                              kernel_initializer=bilinear_interp_init)(net)
    s_tconv2_output = tf.keras.layers.Conv2D(TCONV_OUTPUT_CHANNELS, (1,1), padding='same', activation='relu')(s_tconv2)


    c_fuse = tf.add_n([c_tconv1_output, c_tconv2_output])
    s_fuse = tf.add_n([s_tconv1_output, s_tconv2_output])
    
    tf.logging.info(c_fuse)
    tf.logging.info(s_fuse)
    