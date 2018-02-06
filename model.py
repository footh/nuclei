import tensorflow as tf
from tensorflow.contrib import slim as slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.python.ops import variable_scope

import numpy as np
import math

FEATURE_ROOT = 64

TCONV_ROOT = 8
DROPOUT_PROB = 0.5

tf.logging.set_verbosity(tf.logging.DEBUG)

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
    
    tf.logging.debug(bilinear.shape)        
    weights = np.zeros(shape, dtype=dtype.as_numpy_dtype())
    for i in range(shape[2]):
        weights[:,:,i,i] = bilinear
        
    tf.logging.debug(weights.shape)
    return weights
    #return tf.convert_to_tensor(weights, dtype=dtype, name='bilinear_interp_init')
    #return tf.constant(weights, shape=shape, dtype=dtype, name='bilinear_interp_init')

def build_custom(img_input):
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
    tf.logging.debug(f"net after #3: {net}")
    
    upsample_convs.append(net)
    
    # 4
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*8, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.Dropout(rate=DROPOUT_PROB)(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)
    tf.logging.debug(f"net after #4: {net}")

    upsample_convs.append(net)

    # 5
    net = tf.keras.layers.Conv2D(FEATURE_ROOT*16, (3,3), padding='same', activation='relu')(net)
    net = tf.keras.layers.Dropout(rate=DROPOUT_PROB)(net)
    net = tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2), padding='same')(net)
    tf.logging.debug(f"net after #5: {net}")

    upsample_convs.append(net)

    return upsample_convs

def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v1_50'):
  """
      ResNet-50 model of [1]. See resnet_v1() for arg and return description.
      (same as what's in slim library now but reversing the 1 stride to accommodate the dcan model)
  """
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
    
def build_resnet50_v1(img_input, scope=None):
    """
        Builds resnet50_v1 model from slim, with strides reversed.
        
        Returns the last three block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        block4, endpoints = resnet_v1_50(img_input, is_training=True, global_pool=False)

    prefix = 'resnet_v1_50'
    if scope is not None:
        prefix = f"{scope}/{prefix}"
        
    block3 = endpoints[f"{prefix}/block3"]
    block2 = endpoints[f"{prefix}/block2"]
    
    return block2, block3, block4

def build_resnet50_v2(img_input, scope=None):
    """
        Builds resnet50_v2 model from slim
        
        Returns the last three block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        block2, endpoints = resnet_v2.resnet_v2_50(img_input, is_training=True, global_pool=False)

    prefix = 'resnet_v2_50'
    if scope is not None:
        prefix = f"{scope}/{prefix}"
        
    block3 = endpoints[f"{prefix}/block3"]
    block2 = endpoints[f"{prefix}/block2"]

    return block2, block3, block4

def upsample_and_fuse(ds_layers, img_size):
    """
        Takes in a collection of downsampled layers, applies two transposed convolutions for each input layer and
        fuses each path into one
        
        Returns the two fused layers, one for contours and one for segments
    """
    # TODO: figure out strategy behind kernel sizes
    # TODO: do a 1x1 convolution after t-convolution? The dcan implements it but I think that is based on the referenced
    # FCN paper and implementation. The dcan paper doesn't mention it explicitly.
    # TODO: weight initialization on t-convolution layers, ie bilinear upsampling. The impl above I think is flawed.
    # TODO: regularization? The dcan paper has L2 in the formula. What about dropout? Slim's resnet I believe has L2, need to check
    
    segment_outputs = []
    contour_outputs = []
    
    for i, ds_layer in enumerate(ds_layers):
        kernel = TCONV_ROOT * 2**(i+1)
        stride = img_size // ds_layer.shape.as_list()[1]
        tf.logging.debug(f"layer {i+1} kernel, stride: {kernel, stride}")
        
        net = layers.conv2d_transpose(ds_layer, 
                                      1, 
                                      kernel, 
                                      stride, 
                                      padding='SAME', 
                                      activation_fn=None, 
                                      scope=f"tconv{i+1}_seg")
        segment_outputs.append(net)

        net = layers.conv2d_transpose(ds_layer,
                                      1, 
                                      kernel, 
                                      stride, 
                                      padding='SAME', 
                                      activation_fn=None, 
                                      scope=f"tconv{i+1}_con")
        contour_outputs.append(net)

    
    s_fuse = tf.add_n(segment_outputs, name="segment_fuse")
    c_fuse = tf.add_n(contour_outputs, name="contour_fuse")
    
    return s_fuse, c_fuse

def logits(input, ds_model='resnet50_v1', scope=None):
    """
        Returns the contour and segment logits based on the chosen downsample model. Defaults to 'resnet50_v1'
    """
    img_size = input.shape.as_list()[1]
    if img_size != input.shape.as_list()[2]:
        raise ValueError("Image input must have equal dimensions")
    
    ds_layers = build_resnet50_v1(input, scope=scope)
    return upsample_and_fuse(ds_layers, img_size)
    
    
    