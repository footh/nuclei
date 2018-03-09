import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib import layers
from tensorflow.python.ops import init_ops

import numpy as np

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
      is_training,
      global_pool,
      output_stride,
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
    
    return block2, block3, block4


def build_resnet50_v2(img_input, l2_weight_decay=0.01, is_training=True):
    """
        Builds resnet50_v2 model from slim
        
        Returns the last three block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v2.resnet_v2_50(img_input, is_training=is_training, global_pool=False)

    block3 = endpoints['resnet_v2_50/block3']
    block2 = endpoints['resnet_v2_50/block2']

    return block2, block3, block4


def upsample_aft(ds_layers, img_size):
    """
        Takes in a collection of downsampled layers, applies two transposed convolutions for each input layer returns
        the results. A 1x1 convolution is added after the upsample
        
        Returns the upsampled layers for segments and contours as separate arrays
        
        kernel size calculated per here:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/        

        TODO: bilinear upsampling init? This would require a conv->tconv or tconv->conv where the tconv keeps the channels
        the same and the conv adjusts to the proper channels.
        TODO: regularization? The dcan paper has L2 in the formula. What about dropout? Slim's resnet I believe has L2, need to check
    """
    
    segment_outputs = []
    contour_outputs = []
    
    for i, ds_layer in enumerate(ds_layers):
        factor = img_size // ds_layer.shape.as_list()[1]
        kernel = 2 * factor - factor % 2

        tf.logging.debug(f"layer {i+1} kernel, stride (factor): {kernel, factor}")
        tf.logging.info(f"Layer shape: {ds_layer.shape.as_list()}")

        # Default xavier_initializer is used for the weights here.
        # TODO: this is uniform, should use gaussian per dcan paper?
        net = layers.conv2d_transpose(ds_layer, 
                                      1, 
                                      kernel, 
                                      factor, 
                                      padding='SAME', 
                                      activation_fn=tf.nn.relu,
                                      scope=f"tconv{i+1}_seg")
        net = layers.conv2d(net, 1, 1, activation_fn=None, scope=f"conv{i+1}_seg")

        segment_outputs.append(net)

        net = layers.conv2d_transpose(ds_layer,
                                      1, 
                                      kernel, 
                                      factor,
                                      padding='SAME', 
                                      activation_fn=tf.nn.relu,
                                      scope=f"tconv{i+1}_con")
        net = layers.conv2d(net, 1, 1, activation_fn=None, scope=f"conv{i+1}_con")

        contour_outputs.append(net)
    
    return segment_outputs, contour_outputs


def upsample(ds_layers, img_size):
    """
        Takes in a collection of downsampled layers, applies two transposed convolutions for each input layer returns
        the results. A 1x1 convolution is performed before the transposed convolution
        
        Returns the upsampled layers for segments and contours as separate arrays
        
        kernel size calculated per here:
        http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/        

        TODO: bilinear upsampling init? This would require a conv->tconv or tconv->conv where the tconv keeps the channels
        the same and the conv adjusts to the proper channels.
    """
    
    segment_outputs = []
    contour_outputs = []
    
    for i, ds_layer in enumerate(ds_layers):
        factor = img_size // ds_layer.shape.as_list()[1]
        kernel = 2 * factor - factor % 2

        tf.logging.debug(f"layer {i+1} kernel, stride (factor): {kernel, factor}")
        tf.logging.info(f"Layer shape: {ds_layer.shape.as_list()}")

        # Default xavier_initializer is used for the weights here.
        # TODO: this is uniform, should use gaussian per dcan paper?
        net = layers.conv2d(ds_layer, ds_layer.shape.as_list()[-1], 1, activation_fn=tf.nn.relu, scope=f"conv{i+1}_seg")
        net = layers.conv2d_transpose(net, 
                                      1, 
                                      kernel, 
                                      factor, 
                                      padding='SAME', 
                                      activation_fn=None,
                                      scope=f"tconv{i+1}_seg")

        segment_outputs.append(net)

        net = layers.conv2d(ds_layer, ds_layer.shape.as_list()[-1], 1, activation_fn=tf.nn.relu, scope=f"conv{i+1}_con")
        net = layers.conv2d_transpose(net,
                                      1, 
                                      kernel, 
                                      factor,
                                      padding='SAME', 
                                      activation_fn=None,
                                      scope=f"tconv{i+1}_con")

        contour_outputs.append(net)
    
    return segment_outputs, contour_outputs


def logits(input, ds_model='resnet50_v1', scope='dcan', is_training=True, l2_weight_decay=0.01):
    """
        Returns the contour and segment logits based on the chosen downsample model. Defaults to 'resnet50_v1'
    """

    img_size = input.shape.as_list()[1]
    if img_size != input.shape.as_list()[2]:
        raise ValueError("Image input must have equal dimensions")

    if ds_model == 'resnet50_v1':
        ds_layers = build_resnet50_v1(input, l2_weight_decay=l2_weight_decay, is_training=is_training)
    elif ds_model == 'custom':
        ds_layers = build_custom(input, l2_weight_decay=l2_weight_decay, is_training=is_training)

    with tf.variable_scope(f"{scope}/upsample"), slim.arg_scope([layers.conv2d_transpose], 
                                                                weights_regularizer=slim.l2_regularizer(l2_weight_decay)):
        with slim.arg_scope([layers.conv2d], 
                            normalizer_fn=slim.batch_norm):

            with slim.arg_scope([slim.batch_norm], 
                                scale=True, is_training=is_training):
            
                segment_outputs, contour_outputs = upsample(ds_layers, img_size)
                fuse_seg = tf.add_n(segment_outputs, name="fuse_seg")
                fuse_con = tf.add_n(contour_outputs, name="fuse_con")

    return fuse_seg, fuse_con
