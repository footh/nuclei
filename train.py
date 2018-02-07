import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np

import model
import data

IMG_SIZE = 256
MODEL_SCOPE = "dcan"

# TODO: parameterize these
TRAINING_STEPS = [10000, 5000, 5000]
LEARNING_RATES = [0.001, 0.0005, 0.0001]
BATCH_SIZE = 50
VALIDATION_PCT = 15
VAL_INTERVAL = 400
TRAIN_BASE_DIR = 'training-runs'
L2_WEIGHT_DECAY  = 0.0001
DS_MODEL = 'resnet50_v1'


def loss(logits_seg, logits_con, labels_seg, labels_con):
    """
        Derives a loss function given the segment and contour results and labels
    """
    # TODO: weighting of loss terms. DCAN project uses proportion of values that equal 1 for contours and segments
    # The DCAN paper talks about weights for the 'auxiliary classifiers'. In the FCN which the paper refers, the 
    # auxiliary classifiers are the pre-fused results from the different levels of the convnet. Should that be the same here?
    # Means there will be 6 of them - 3 for each label type. FCN code doesn't reveal much about weighting and the paper doesn't 
    # help much either.
    
    s_loss = tf.losses.sigmoid_cross_entropy(labels_seg, logits_seg, scope='segment_loss')
    c_loss = tf.losses.sigmoid_cross_entropy(labels_con, logits_con, scope='contour_loss')
    
    total_loss = tf.add(s_loss, c_loss, name='total_loss')
    
    return total_loss


def _get_learning_rate(training_step):
    """
        Return the learning rate based on the current step
    """
    training_step_bucket = 0
    for i in range(len(TRAINING_STEPS)):
        training_step_bucket += TRAINING_STEPS[i]
        if training_step <= training_step_bucket:
            return LEARNING_RATES[i]


def _get_train_dir():
    """
        Returns the train directory, makes the directory if it doesn't exist
    """
    cur_train_dir = os.path.join(TRAIN_BASE_DIR, 'tbd')
    tf.gfile.MakeDirs(cur_train_dir)
    return cur_train_dir


def _restore_from_checkpoint(model_path, sess, var_filter=None):
    """
        Restores variables from given checkpoint file, variables can be filterd by 'var_filter' argument
    """
    vars_to_restore = slim.get_model_variables(var_filter)
    restore_fn = slim.assign_from_checkpoint_fn(model_path, vars_to_restore, ignore_missing_vars=True)

    restore_fn(sess)


def _get_trainable_vars():
    pass


def train():
    train_dir = _get_train_dir()
    
    sess = tf.InteractiveSession()
    
    data_processor = data.DataProcessor(img_size=IMG_SIZE, validation_pct=VALIDATION_PCT)
    
    with tf.variable_scope(MODEL_SCOPE):
    
        img_input = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='img_input')
        labels_seg = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='s_labels')
        labels_con = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='c_labels')
    
    logits_seg, logits_con = model.logits(img_input, scope=MODEL_SCOPE, l2_weight_decay=L2_WEIGHT_DECAY)
        
    with tf.variable_scope(MODEL_SCOPE):

        logits_seg = tf.squeeze(logits_seg, axis=-1)
        logits_con = tf.squeeze(logits_con, axis=-1)

        total_loss = loss(logits_seg, logits_con, labels_seg, labels_con)
        tf.summary.scalar('total_loss', total_loss)

    with tf.name_scope('train'), tf.control_dependencies(tf.GraphKeys.UPDATE_OPS):

        lr_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
        train_op = tf.train.AdamOptimizer(lr_input).minimize(total_loss)

    # TODO: add accuracy calculation (IOU?, standard accuracy?) here. This will be added to the session run below
    
    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    
    # Global variable saver
    saver = tf.train.Saver()
    
    # Run an initializer op to initialize the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Log the number of parameters
    params = tf.trainable_variables()
    num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    tf.logging.info(f"Total number or trainable parameters: {num_params}")
    
    start_step = 1
    
    # TODO: parameterize
    model_path = None
    if model_path is not None:
        _restore_from_checkpoint(model_path, sess, var_filter='resnet_v1_50')
    
    tf.logging.info('Training from step: %d ', start_step)
    
    # Save graph
    # TODO: more meaningful name (with tweaked parameters?)
    tf.train.write_graph(sess.graph_def, train_dir, f"{MODEL_SCOPE}-{DS_MODEL}.pbtxt")

    # Training loop -------------------------
    best_valid_loss = 0
    max_training_steps = np.sum(TRAINING_STEPS)
    for training_step in range(start_step, max_training_steps + 1):
        
        learning_rate = _get_learning_rate(training_step)
    
        x, y_seg, y_con = data_processor.batch(BATCH_SIZE, offset=0, mode='train')
        
        train_loss, _, _ = sess.run([total_loss, train_op, increment_global_step],
                                    feed_dict={img_input: x, 
                                               labels_seg: y_seg, 
                                               labels_con: y_con, 
                                               lr_input: learning_rate})

        tf.logging.info(f"Step {training_step}: rate {learning_rate}, accuracy TBD, loss {train_loss}")
    
        if (training_step % VAL_INTERVAL) == 0 or (training_step == training_steps_max):
            val_size = data_processor.mode_size(mode='valid')
            valid_loss = 0
            
            for val_step in range(0, val_size, BATCH_SIZE):
                val_x, val_y_seg, val_y_con = data_processor.batch(BATCH_SIZE, offset=val_step, mode='valid')
    
                batch_valid_loss = sess.run([total_loss],
                                            feed_dict={img_input: val_x,
                                                       labels_seg: val_y_seg, 
                                                       labels_con: val_y_con})
    
                valid_loss += (batch_valid_loss * val_x.shape[0]) / val_size
    
            tf.logging.info(f"Step {training_step}: validation loss = {valid_loss}, (N={val_size}")
    
            # Save the model checkpoint when validation loss improves
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                checkpoint_path = os.path.join(train_dir, 'best', f"{MODEL_SCOPE}_val-loss-{valid_loss:.5f}.ckpt")
                tf.logging.info(f"Saving best model to {checkpoint_path}-{training_step}")
                saver.save(sess, checkpoint_path, global_step=training_step)

            tf.logging.info(f"Best validation loss so far: {best_valid_loss}")
