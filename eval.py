import tensorflow as tf
import train
import data
import model
import numpy as np


def evaluate():
    sess = tf.InteractiveSession()

    data_processor = data.DataProcessor(src='test', img_size=train.IMG_SIZE, testing_pct=100)

    with tf.variable_scope(f"{train.MODEL_SCOPE}/data"):
        img_input = tf.placeholder(tf.float32, [None, train.IMG_SIZE, train.IMG_SIZE, 3], name='img_input')

    # TODO: parameterize the ds_model
    logits_seg, logits_con = model.logits(img_input,
                                          ds_model='resnet50_v1',
                                          scope=train.MODEL_SCOPE,
                                          is_training=False)

    with tf.variable_scope(f"{train.MODEL_SCOPE}/prediction"):
        pred_seg = tf.nn.sigmoid(logits_seg, name='pred_seg')
        pred_con = tf.nn.sigmoid(logits_con, name='pred_con')

        pred_seg = tf.squeeze(pred_seg, axis=-1, name='squeeze_seg')
        pred_con = tf.squeeze(pred_con, axis=-1, name='squeeze_con')

        pred_full = tf.stack([pred_seg, pred_con], axis=-1)

    train.restore_from_checkpoint(FLAGS.trained_checkpoint, sess)

    test_batch = data_processor.batch_test(offset=0)
    test_batch = test_batch.reshape(test_batch.shape[0] * test_batch.shape[1], *test_batch.shape[2:])

    test_pred = sess.run(pred_full, feed_dict={img_input: test_batch})
    tf.logging.info(f"test_pred.shape: {test_pred.shape}")
    np.save('test_pred.npy', test_pred)
    

def main(_):
    if not FLAGS.trained_checkpoint:
        raise ValueError('A trained checkpoint must be set for evaluation')

    tf.logging.set_verbosity(tf.logging.INFO)

    evaluate()


tf.app.flags.DEFINE_string(
    'trained_checkpoint', None,
    'The checkpoint to initialize evaluation from')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()
