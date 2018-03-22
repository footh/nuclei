import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops

import numpy as np
from mpmath import residual

DROPOUT_PROB = 0.5

#tf.logging.set_verbosity(tf.logging.DEBUG)


class BilinearInterp(init_ops.Initializer):
    """
        Bilinear interpolation initializer. Initializes a t-conv weight matrix for bilinear upsampling per this link:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/        
    """
    def __init__(self, dtype=tf.float32):
        self.dtype = init_ops._assert_float_dtype(tf.as_dtype(dtype))

    def __call__(self, shape, dtype=None, partition_info=None):
        if shape[0] != shape[1]:
            raise ValueError("Bilinear interp init must have equal kernel dimensions")

        if shape[2] != shape[3]:
            raise ValueError("Bilinear interp init must have equal channel dimensions")
        
        if dtype is None:
            dtype = self.dtype

        upsample_kernel = self.upsample_filt(shape[0])
        
        weights = np.zeros(shape, dtype=self.dtype.as_numpy_dtype())
        for i in range(shape[2]):
            weights[:, :, i, i] = upsample_kernel
            
        return tf.convert_to_tensor(weights, dtype=dtype, name='bilinear_interp_init')

    def get_config(self):
        return {"dtype": self.dtype.name}
        
    def upsample_filt(self, size):
        """
            Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)        


bilinear_interp = BilinearInterp


def custom_arg_scope(weight_decay=0.01,
                     batch_norm_decay=0.997,
                     batch_norm_epsilon=1e-5,
                     batch_norm_scale=True,
                     activation_fn=tf.nn.relu,
                     use_batch_norm=True):

    batch_norm_params = {
       'decay': batch_norm_decay,
       'epsilon': batch_norm_epsilon,
       'scale': batch_norm_scale,
       'updates_collections': tf.GraphKeys.UPDATE_OPS,
       'fused': None,  # Use fused batch norm if possible.
    }
    
    with slim.arg_scope([slim.conv2d],
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=slim.variance_scaling_initializer(),
                        activation_fn=activation_fn,
                        normalizer_fn=slim.batch_norm if use_batch_norm else None,
                        normalizer_params=batch_norm_params):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
            with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc    


def build_custom(inputs, l2_weight_decay=0.01, is_training=True):
    tf.logging.info("CUSTOM")
    feature_root = 64
    
    blocks = []
    with slim.arg_scope(custom_arg_scope(weight_decay=l2_weight_decay)):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = slim.conv2d(inputs, feature_root, (3, 3))
            
            net = slim.conv2d(net, feature_root * 2, (3, 3))
            net = slim.max_pool2d(net, (2, 2), stride=(2, 2))

            net = slim.conv2d(net, feature_root * 4, (3, 3))
            net = slim.max_pool2d(net, (2, 2), stride=(2, 2))
        
            net = slim.conv2d(net, feature_root * 8, (3, 3))
            net = slim.max_pool2d(net, (2, 2), stride=(2, 2))
            
            blocks.append(net)

            net = slim.conv2d(net, feature_root * 16, (3, 3))
            net = slim.max_pool2d(net, (2, 2), stride=(2, 2))

            blocks.append(net)

            net = slim.conv2d(net, feature_root * 32, (3, 3))
            net = slim.max_pool2d(net, (2, 2), stride=(2, 2))
            
            blocks.append(net)
    
    return blocks


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
      is_training=is_training,
      global_pool=global_pool,
      output_stride=output_stride,
      include_root_block=True,
      reuse=reuse,
      scope=scope)


def build_resnet50_v1(img_input, l2_weight_decay=0.01, is_training=True):
    """
        Builds resnet50_v1 model from slim, with strides reversed.
        
        Returns the last three block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v1_50(img_input, is_training=is_training, global_pool=False)

    block3 = endpoints['resnet_v1_50/block3']
    block2 = endpoints['resnet_v1_50/block2']
    block1 = endpoints['resnet_v1_50/block1']
    
    return block1, block2, block3, block4
    #return block2, block3, block4


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
    """
        ResNet-50 model of [1]. See resnet_v2() for arg and return description.
    """
    blocks = [
        resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=3, stride=1),
        resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2.resnet_v2_block('block4', base_depth=512, num_units=3, stride=2),
    ]

    return resnet_v2.resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)


def build_resnet50_v2(img_input, l2_weight_decay=0.01, is_training=True):
    """
        Builds resnet50_v2 model from slim
        
        Returns the last three block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v2_50(img_input, is_training=is_training, global_pool=False)

    block3 = endpoints['resnet_v2_50/block3']
    block2 = endpoints['resnet_v2_50/block2']
    block1 = endpoints['resnet_v2_50/block1']

    return block1, block2, block3, block4


def upsample(ds_layers, img_size, type='seg'):
    """
        Takes in a collection of downsampled layers, applies  transposed convolutions for each input layer returns
        the results. A 1x1 convolution is performed before the transposed convolution
        
        Returns the upsampled layers as an array
        
        kernel size calculated per here:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/        

        TODO: bilinear upsampling init? This would require a conv->tconv or tconv->conv where the tconv keeps the channels
        the same and the conv adjusts to the proper channels.
    """
    upsampled_outputs = []
    
    for i, ds_layer in enumerate(ds_layers):
        factor = img_size // ds_layer.shape.as_list()[1]
        kernel = 2 * factor - factor % 2

        tf.logging.debug(f"layer {i+1} kernel, stride (factor): {kernel, factor}")
        tf.logging.info(f"Layer shape: {ds_layer.shape.as_list()}")

        # Default xavier_initializer is used for the weights here.
        # TODO: this is uniform, should use gaussian per dcan paper?
        #net = layers.conv2d(ds_layer, ds_layer.shape.as_list()[-1], 1, activation_fn=tf.nn.relu, scope=f"conv{i+1}_{type}")
        #net = layers.conv2d(net, net.shape.as_list()[-1], 3, activation_fn=tf.nn.relu, scope=f"convb{i+1}_{type}")
        net = layers.conv2d_transpose(ds_layer, 
                                      1, 
                                      kernel, 
                                      factor, 
                                      padding='SAME', 
                                      activation_fn=None,
                                      scope=f"tconv{i+1}_{type}")

        upsampled_outputs.append(net)
    
    return upsampled_outputs   

def residual_ds_layers(ds_layers, type='seg'):
    """
        Perform a residual block on each incoming downsampled layer
    """
    residual_output = []
    
    for i, ds_layer in enumerate(ds_layers):
        net = layers.conv2d(ds_layer, ds_layer.shape.as_list()[-1], 3, activation_fn=tf.nn.relu, scope=f"conv_a{i+1}_{type}")
        net = layers.conv2d(net, net.shape.as_list()[-1], 3, activation_fn=tf.nn.relu, scope=f"conv_b{i+1}_{type}")
        
        net = tf.nn.relu(tf.add(ds_layer, net, name=f"fuse_{type}"))
        residual_output.append(net)

    return residual_output

def process_ds_layers(ds_layers, channels_out=256, type='seg'):
    """
        Process the downsample layers by running a convolution to get to desired output channels and upsampling each
        deeper layer to fuse with layer right above. Fused layers get a final 3x3 convolution.
    """
    ds_layers_out = []
            
    # NOTE: default activation for conv2d is tf.nn.relu
    # NOTE: default uniform xavier_initializer is used for the weights here.
    index = len(ds_layers) - 1
    while index >= 0:
        tf.logging.info(ds_layers[index].shape.as_list())
        net = layers.conv2d(ds_layers[index], channels_out, 1, padding='VALID', scope=f"{type}_conv{index+1}")
        if len(ds_layers_out) > 0:
            up_layer = ds_layers_out[-1]
            up_size = [2 * l for l in up_layer.shape.as_list()[1:3]]
            up = tf.image.resize_images(up_layer, up_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            
            net = tf.add(net, up, name=f"{type}_fuse{index+1}")
            net = layers.conv2d(net, channels_out, 3, scope=f"{type}_fuseconv{index+1}")
        
        ds_layers_out.append(net)
        index -= 1

    return ds_layers_out[::-1]


def attention_ds_layers(ds_layers):
    
    seg_layers = []
    con_layers = []
    for i, ds_layer in enumerate(ds_layers):
        avg = slim.layers.avg_pool2d(ds_layer, ds_layer.shape.as_list()[1:3], stride=1, scope="avg_pool")
        avg = tf.squeeze(avg, axis=(1,2))
        tf.logging.info(f"avg pool shape: {avg.shape.as_list()}")
        
        attn_seg = tf.layers.dense(avg, avg.shape.as_list()[-1], kernel_initializer=slim.xavier_initializer(), name=f"seg_weights{i+1}")
        attn_seg = tf.expand_dims(attn_seg, 1)
        attn_seg = tf.expand_dims(attn_seg, 1)
        tf.logging.info(f"attn_seg weights shape: {attn_seg.shape.as_list()}")
        attn_seg = tf.multiply(ds_layer, attn_seg, name=f"seg_weighted{i+1}")
        tf.logging.info(f"attn seg out shape: {attn_seg.shape.as_list()}")
        seg_layers.append(attn_seg)
        
        attn_con = tf.layers.dense(avg, avg.shape.as_list()[-1], kernel_initializer=slim.xavier_initializer(), name=f"con_weights{i+1}")
        attn_con = tf.expand_dims(attn_con, 1)
        attn_con = tf.expand_dims(attn_con, 1)
        tf.logging.info(f"attn con weights shape: {attn_con.shape.as_list()}")
        attn_con = tf.multiply(ds_layer, attn_con, name=f"con_weighted{i+1}")
        tf.logging.info(f"attn con out shape: {attn_con.shape.as_list()}")
        con_layers.append(attn_con)
        
    return seg_layers, con_layers


def logits(input, ds_model='resnet50_v1', scope='dcan', is_training=True, l2_weight_decay=0.01):
    """
        Returns the contour and segment logits based on the chosen downsample model. Defaults to 'resnet50_v1'
    """

    img_size = input.shape.as_list()[1]
    if img_size != input.shape.as_list()[2]:
        raise ValueError("Image input must have equal dimensions")

    # Extract features from downsampling net
    if ds_model == 'resnet50_v1':
        ds_layers = build_resnet50_v1(input, l2_weight_decay=l2_weight_decay, is_training=is_training)
    elif ds_model == 'resnet50_v2':
        ds_layers = build_resnet50_v2(input, l2_weight_decay=l2_weight_decay, is_training=is_training)
    elif ds_model == 'custom':
        ds_layers = build_custom(input, l2_weight_decay=l2_weight_decay, is_training=is_training)
        
#     # Add attention to each downsampled layer for both segments and contours
#     with tf.variable_scope(f"{scope}/attn"):
#         seg_layers, con_layers = attention_ds_layers(ds_layers)

    # Process extracted downsampled layers
    with tf.variable_scope(f"{scope}/process_ds"), slim.arg_scope([layers.conv2d],
                                                                  weights_regularizer=slim.l2_regularizer(l2_weight_decay)):
        ds_layers = process_ds_layers(ds_layers)
        #seg_layers = process_ds_layers(ds_layers, type='seg')
        #con_layers = process_ds_layers(ds_layers, type='con')

    with tf.variable_scope(f"{scope}/residual_ds"), slim.arg_scope([layers.conv2d],
                                                                   weights_regularizer=slim.l2_regularizer(l2_weight_decay),
                                                                   normalizer_fn=None):
        #with slim.arg_scope([slim.batch_norm], is_training=is_training, scale=True):
        seg_layers = residual_ds_layers(ds_layers, type='seg')
        con_layers = residual_ds_layers(ds_layers, type='con')


    # Upsample to image size the processed ds layers to segment and contour results
    with tf.variable_scope(f"{scope}/upsample"), slim.arg_scope([layers.conv2d_transpose], 
                                                                weights_regularizer=slim.l2_regularizer(l2_weight_decay)):
        segment_outputs = upsample(seg_layers, img_size, type='seg')
        contour_outputs = upsample(con_layers, img_size, type='con')
        #segment_outputs = upsample(ds_layers, img_size, type='seg')
        #contour_outputs = upsample(ds_layers, img_size, type='con')

    # Fuse the segment and contour results
    fuse_seg = tf.add_n(segment_outputs, name="fuse_seg")
    fuse_con = tf.add_n(contour_outputs, name="fuse_con")

    return fuse_seg, fuse_con
