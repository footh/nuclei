import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os

import model
import data

IMG_SIZE = 256
MODEL_SCOPE = "dcan"

# TODO: parameterize these
TRAINING_STEPS = [8000, 8000, 8000, 8000]

LEARNING_RATES = [0.005, 0.001, 0.0007, 0.0003]
ADAM_EPSILON = 1e-08

VALID_LOSS_STREAK_MAX = 10
EXPONENTIAL_DECAY_BASE = 0.5
LEARNING_RATE_BASE = 0.001
MOMENTUM = 0.2

VALIDATION_PCT = 15
VAL_INTERVAL = 300
TRAIN_BASE_DIR = 'training-runs'
L2_WEIGHT_DECAY = 0.0001
#SEG_RATIO = 0.13405
#CON_RATIO = 0.04466


def loss(logits_seg, logits_con, labels_seg, labels_con):
    """
        Derives a loss function given the segment and contour results and labels
    """
    # TODO: weighting of loss terms. DCAN project uses proportion of values that equal 1 for contours and segments
    # The DCAN paper talks about weights for the 'auxiliary classifiers'. In the FCN which the paper refers, the 
    # auxiliary classifiers are the pre-fused results from the different levels of the convnet. Should that be the same here?
    # Means there will be 6 of them - 3 for each label type. FCN code doesn't reveal much about weighting and the paper doesn't 
    # help much either.
    
#     with tf.variable_scope("weights/segment"):
#         weights_seg = tf.scalar_mul(SEG_RATIO, tf.cast(tf.equal(labels_seg, 0), tf.float32)) + \
#                       tf.scalar_mul(1 - SEG_RATIO, tf.cast(tf.equal(labels_seg, 1), tf.float32))
# 
#     with tf.variable_scope("weights/contour"):
#         weights_con = tf.scalar_mul(CON_RATIO, tf.cast(tf.equal(labels_con, 0), tf.float32)) + \
#                       tf.scalar_mul(1 - CON_RATIO, tf.cast(tf.equal(labels_con, 1), tf.float32))
#     
#     loss_seg = tf.losses.sigmoid_cross_entropy(labels_seg, logits_seg, weights=weights_seg, scope='segment_loss')
#     loss_con = tf.losses.sigmoid_cross_entropy(labels_con, logits_con, weights=weights_con, scope='contour_loss')

    loss_seg = tf.losses.sigmoid_cross_entropy(labels_seg, logits_seg, scope='segment_loss')
    loss_con = tf.losses.sigmoid_cross_entropy(labels_con, logits_con, scope='contour_loss')

    total_loss = tf.add(loss_seg, loss_con, name='total_loss')
    
    return total_loss


def iou(logits, labels, scope=None):
    
    with tf.variable_scope(f"iou/{scope}"):
        preds = tf.nn.sigmoid(logits)
        preds = tf.greater_equal(preds, 0.5)
        labels_bool = tf.cast(labels, tf.bool)
        
        intersection = tf.reduce_sum(tf.cast(tf.logical_and(preds, labels_bool), tf.float32))
        union = tf.reduce_sum(tf.cast(tf.logical_or(preds, labels_bool), tf.float32))
    
    return intersection / union


def restore_from_checkpoint(model_path, sess, var_filter=None):
    """
        Restores variables from given checkpoint file, variables can be filtered by 'var_filter' argument
    """
    vars_to_restore = slim.get_model_variables(var_filter)
    restore_fn = slim.assign_from_checkpoint_fn(model_path, vars_to_restore, ignore_missing_vars=True)

    restore_fn(sess)
    tf.logging.info(f"Successfully restored checkpoint using path: {model_path} with filter: {var_filter}")


def _get_learning_rate(training_step):
    """
        Return the learning rate based on the current step
    """
    training_step_bucket = 0
    for i in range(len(TRAINING_STEPS)):
        training_step_bucket += TRAINING_STEPS[i]
        if training_step <= training_step_bucket:
            return LEARNING_RATES[i]


def _get_learning_rate_decay(valid_loss_streak_hits):
    """
        Return the learning rate based on the decay params
    """
    return LEARNING_RATE_BASE * (EXPONENTIAL_DECAY_BASE ** valid_loss_streak_hits)


def _get_train_dir():
    """
        Returns the train directory, makes the directory if it doesn't exist
    """
    cur_train_dir = os.path.join(TRAIN_BASE_DIR, FLAGS.run_desc)
    tf.gfile.MakeDirs(cur_train_dir)
    return cur_train_dir


def _write_notes(train_dir):
    response = input("Training notes? (ENTER skips) ")
    if response is not None:
        desc_file = os.path.join(train_dir, 'notes.txt')
        with open(desc_file, 'w') as f:
            f.write(response)


def _get_trainable_vars():
    pass


def train():
    sess = tf.InteractiveSession()

    if FLAGS.debug_graph:
        from tensorflow.python import debug as tf_debug
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    train_dir = _get_train_dir()
    if FLAGS.notes:
        _write_notes(train_dir)
    
    data_processor = data.DataProcessor(img_size=IMG_SIZE, validation_pct=VALIDATION_PCT)

    with tf.variable_scope(f"{MODEL_SCOPE}/data"):
        is_training = tf.placeholder(tf.bool)

        img_input = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3], name='img_input')
        labels_seg = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='labels_seg')
        labels_con = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE], name='labels_con')
    
    logits_seg, logits_con = model.logits(img_input,
                                          ds_model=FLAGS.ds_model,
                                          scope=MODEL_SCOPE,
                                          is_training=is_training,
                                          l2_weight_decay=L2_WEIGHT_DECAY)
        
    with tf.variable_scope(f"{MODEL_SCOPE}/loss"):

        logits_seg = tf.squeeze(logits_seg, axis=-1, name='squeeze_seg')
        logits_con = tf.squeeze(logits_con, axis=-1, name='squeeze_con')

        total_loss = loss(logits_seg, logits_con, labels_seg, labels_con)
        tf.summary.scalar('total_loss', total_loss)

        # IOU calcs to measure during training
        iou_seg = iou(logits_seg, labels_seg, scope='segment')
        tf.summary.scalar('segment_iou', iou_seg)
        iou_con = iou(logits_con, labels_con, scope='contour')
        tf.summary.scalar('contour_iou', iou_con)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.name_scope('train'), tf.control_dependencies(update_ops):

        lr_input = tf.placeholder(tf.float32, [], name='learning_rate_input')
        # train_op = tf.train.AdamOptimizer(learning_rate=lr_input, epsilon=ADAM_EPSILON).minimize(total_loss)
        train_op = tf.train.MomentumOptimizer(learning_rate=lr_input, momentum=MOMENTUM).minimize(total_loss)

    global_step = tf.train.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)
    
    # Global variable saver
    saver = tf.train.Saver()

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(train_dir, 'summary', 'train'), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(train_dir, 'summary', 'valid'))
    
    # Run an initializer op to initialize the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Log the number of parameters
    #params = tf.trainable_variables()
    #num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
    #tf.logging.info(f"Total number or trainable parameters: {num_params}")
    
    start_step = 1
    
    # Restore any provided checkpoint. Variables initialized above will be overwritten.
    if FLAGS.checkpoint_file is not None:
        restore_from_checkpoint(FLAGS.checkpoint_file, sess, var_filter=FLAGS.checkpoint_filter)
    
    tf.logging.info('Training from step: %d ', start_step)
    
    # Save graph
    tf.train.write_graph(sess.graph_def, train_dir, f"{MODEL_SCOPE}-{FLAGS.ds_model}.pbtxt")
    
    # Training loop -------------------------
    best_valid_loss = 1000.
    valid_loss_streak = 0
    valid_loss_streak_hits = 0
    max_training_steps = np.sum(TRAINING_STEPS)
    for training_step in range(start_step, max_training_steps + 1):
        
        # learning_rate = _get_learning_rate(training_step)
        learning_rate = _get_learning_rate_decay(valid_loss_streak_hits)

        x, y_seg, y_con = data_processor.batch(FLAGS.batch_size, offset=0, mode='train')
        
        train_loss, train_iou_seg, train_iou_con, train_summary, _, _ = sess.run([total_loss, 
                                                                                  iou_seg, 
                                                                                  iou_con, 
                                                                                  merged_summaries, 
                                                                                  train_op, 
                                                                                  increment_global_step], 
                                                                                  feed_dict={img_input: x,
                                                                                             labels_seg: y_seg,
                                                                                             labels_con: y_con, 
                                                                                             lr_input: learning_rate,
                                                                                             is_training: True})
        
        train_writer.add_summary(train_summary, training_step)
        msg = f"Step {training_step}: learning rate {learning_rate:.5f}, IOU segment {train_iou_seg:.3f}, IOU contour {train_iou_con:.3f}, loss {train_loss:.5f}"
        tf.logging.info(msg)
    
        if (training_step % VAL_INTERVAL) == 0 or (training_step == max_training_steps):
            val_size = data_processor.mode_size(mode='valid')
            valid_loss = 0
            valid_iou_seg = 0
            valid_iou_con = 0
            
            for val_step in range(0, val_size, FLAGS.batch_size):
                val_x, val_y_seg, val_y_con = data_processor.batch(FLAGS.batch_size, offset=val_step, mode='valid')
    
                # TODO: When is_training is set to false, the val loss is very odd. Guess I must leave it at true since
                # it's using the graph built with batch norm :shrug:
                batch_valid_loss, batch_valid_iou_seg, batch_valid_iou_con, valid_summary = sess.run([total_loss, iou_seg, iou_con, merged_summaries],
                                                                                                     feed_dict={img_input: val_x,
                                                                                                                labels_seg: val_y_seg, 
                                                                                                                labels_con: val_y_con,
                                                                                                                is_training: True})
    
                valid_loss += (batch_valid_loss * val_x.shape[0]) / val_size
                valid_iou_seg += (batch_valid_iou_seg * val_x.shape[0]) / val_size
                valid_iou_con += (batch_valid_iou_con * val_x.shape[0]) / val_size
                valid_writer.add_summary(valid_summary, training_step)
                
            msg = f"Step {training_step}: validation loss {valid_loss:.5f}, iou segment {valid_iou_seg:.3f}, iou contour {valid_iou_con:.3f} (N={val_size})"
            tf.logging.info(msg)
    
            # Save the model checkpoint when validation loss improves
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                valid_loss_streak = 0
                checkpoint_path = os.path.join(train_dir, 'best', f"{MODEL_SCOPE}_vloss-{valid_loss:.5f}.ckpt")
                tf.logging.info(f"Saving best model to {checkpoint_path}-{training_step}")
                saver.save(sess, checkpoint_path, global_step=training_step)
            else:
                valid_loss_streak += 1
                if valid_loss_streak >= VALID_LOSS_STREAK_MAX:
                    valid_loss_streak = 0
                    valid_loss_streak_hits += 1
                    tf.logging.info(f"Valdation loss has not increased for {VALID_LOSS_STREAK_MAX} steps")
                    tf.logging.info(f"Decay exponent increased to {valid_loss_streak_hits}")

            tf.logging.info(f"Best validation loss so far: {best_valid_loss:.5f}")


def main(_):
    #if not FLAGS.dataset_dir:
    #    raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    
    train()


tf.app.flags.DEFINE_boolean(
    'debug_graph', False,
    'Wrap the training in a debug session')

tf.app.flags.DEFINE_integer(
    'batch_size', 5,
    'Batch size used for training')

tf.app.flags.DEFINE_string(
    'run_desc', 'test_run',
    'Description for the run, used to name save directory. Should be used to distinguish the run.')

tf.app.flags.DEFINE_string(
    'ds_model', 'resnet50_v1',
    'The down-sample model to use')

tf.app.flags.DEFINE_boolean(
    'notes', False,
    'Add notes to to a file in training directory')

# ----------------------
# Checkpoint parameters
# ----------------------
tf.app.flags.DEFINE_string(
    'checkpoint_file', None,
    'The checkpoint to initialize training from')

tf.app.flags.DEFINE_string(
    'checkpoint_filter', None,
    'The checkpoint filter to target initializing variables. Leaving at None initializes all')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()
