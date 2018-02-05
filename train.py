import tensorflow as tf

import model

IMG_SIZE = 256
MODEL_SCOPE = "dcan"

def loss(c_logits, s_logits, c_labels, s_labels):
    """
        Derives a loss function given the contour and segment results and labels
    """
    # TODO: weighting of loss terms. DCAN project uses proportion of values that equal 1 for contours and segments
    # The DCAN paper talks about weights for the 'auxiliary classifiers'. In the FCN which the paper refers, the 
    # auxiliary classifiers are the pre-fused results from the different levels of the convnet. Should that be the same here?
    # Means there will be 6 of them - 3 for each label type. FCN code doesn't reveal much about weighting and the paper doesn't 
    # help much either.
    
    c_loss = tf.losses.sigmoid_cross_entropy(c_labels, c_logits, scope='contour_loss')
    s_loss = tf.losses.sigmoid_cross_entropy(s_labels, s_logits, scope='segment_loss')
    
    total_loss = tf.add(c_loss, s_loss, name='total_loss')
    
    return total_loss

def train():
    sess = tf.InteractiveSession()
    
    with tf.variable_scope(MODEL_SCOPE):
    
        img_input = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='img_input')
        c_labels = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='c_labels')
        s_labels = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='s_labels')
    
        c_logits, s_logits = model.logits(img_input, scope=MODEL_SCOPE)
        
        c_logits = tf.squeeze(c_logits, axis=-1)
        s_logits = tf.squeeze(s_logits, axis=-1)

        total_loss = loss(c_logits, s_logits, c_labels, s_labels)

    return total_loss