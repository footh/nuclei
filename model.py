import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib.slim.python.slim.nets import resnet_v1

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

def build_custom():

    img_input = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='img_input')
  
    # Better way to do this is, build out all the convolutions first and save the last 3 in different variables, then run the upsampling
    # on the three different convolution points. 
    
    upsample_convs = []
    
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
    
    upsample_convs.append(net)
    
    # 4
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*8, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.Dropout(rate=DROPOUT_PROB)(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)
    tf.logging.info(f"net after #4: {net}")

    upsample_convs.append(net)

    # 5
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*16, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.Dropout(rate=DROPOUT_PROB)(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)
    tf.logging.info(f"net after #5: {net}")

    upsample_convs.append(net)


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
    

def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50'):
  """ResNet-50 model of [1]. See resnet_v1() for arg and return description."""
  blocks = [
      resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
      resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
      resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
      resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
  ]
  return resnet_v1.resnet_v1(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)    
    
def build_resnet50_v1():

    img_input = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='img_input')

    #model = tf.keras.applications.ResNet50(include_top=False, input_tensor=img_input)
    
    #model.summary()
    #return model
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        net, endpoints = resnet_v1_50(img_input, is_training=True, global_pool=False)

    return net, endpoints

def build_resnet50_v2():

    img_input = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='img_input')

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, endpoints = resnet_v2.resnet_v2_50(img_input, is_training=True, global_pool=False)

    return net, endpoints

def test():
    img_input = tf.placeholder(tf.float32, [None, IMG_HEIGHT, IMG_WIDTH, 3], name='img_input')
    
    x = tf.keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = tf.keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    return x
    