import tensorflow as tf
import train
import data
import model
import numpy as np
import shutil
import os
from PIL import Image

OVERLAP_CONST = 2


def _debug_output(sample_info, result_seg, result_con):
    fname = f"{sample_info['id']}-{data.IMG_SRC}.{data.IMG_EXT}"
    shutil.copy2(os.path.join('test', fname), os.path.join('debug', fname))
    
    result_seg = np.asarray(result_seg > 0.5, dtype=np.uint8) * 255
    result_con = np.asarray(result_con > 0.2, dtype=np.uint8) * 255
    Image.fromarray(result_seg).save(os.path.join('debug', fname.replace(data.IMG_SRC, data.IMG_SEGMENT)))
    Image.fromarray(result_con).save(os.path.join('debug', fname.replace(data.IMG_SRC, data.IMG_CONTOUR)))


def evaluate(trained_checkpoint, pixel_threshold=0.5, contour_threshold=0.5):
    # TODO: parameterize
    window_size = train.IMG_SIZE
    
    sess = tf.InteractiveSession()

    with tf.variable_scope(f"{train.MODEL_SCOPE}/data"):
        img_input = tf.placeholder(tf.float32, [None, window_size, window_size, 3], name='img_input')

    # Build model -------------------------------
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

    train.restore_from_checkpoint(trained_checkpoint, sess)
    # -------------------------------------------

    data_processor = data.DataProcessor(src='test', img_size=window_size, testing_pct=100)

    for cnt in range(5):
        sample_tiles, sample_info = data_processor.batch_test(offset=cnt, overlap_const=OVERLAP_CONST)
    
        # Prediction --------------------------------
        tile_rows, tile_cols = sample_tiles.shape[0:2]
        tf.logging.info(f"tile_rows, tile_cols: {tile_rows}, {tile_cols}")
        
        # TODO: with large number of tiles, may need to batch this further
        sample_batch = sample_tiles.reshape(tile_rows * tile_cols, *sample_tiles.shape[2:])
    
        sample_pred = sess.run(pred_full, feed_dict={img_input: sample_batch})
        tf.logging.info(f"sample_pred.shape: {sample_pred.shape}")
    
        sample_pred = sample_pred.reshape(tile_rows, tile_cols, *sample_pred.shape[1:])
        # -------------------------------------------
    
        step = sample_info['step']
        full_pred_rows = (tile_rows + 1) * step
        full_pred_cols = (tile_cols + 1) * step
        tf.logging.info(f"full_pred_rows, full_pred_cols: {full_pred_rows}, {full_pred_cols}")
        
        result_seg = np.zeros((full_pred_rows, full_pred_cols), dtype=np.float32)
        result_con = np.zeros((full_pred_rows, full_pred_cols), dtype=np.float32)
        divisors = np.zeros((full_pred_rows, full_pred_cols), dtype=np.float32)
    
        for i, row_start in enumerate(range(0, full_pred_rows - step, step)):
            for j, col_start in enumerate(range(0, full_pred_cols - step, step)):
                seg = sample_pred[i, j, :, :, 0]
                con = sample_pred[i, j, :, :, 1]
                result_seg[row_start:row_start + window_size, col_start:col_start + window_size] += seg
                result_con[row_start:row_start + window_size, col_start:col_start + window_size] += con
                divisors[row_start:row_start + window_size, col_start:col_start + window_size] += 1.
                
        result_seg = result_seg / divisors
        result_con = result_con / divisors
        
        padding = sample_info['padding']
        orig_rows, orig_cols = sample_info['orig_shape'][0:2]
        result_seg = result_seg[padding:padding + orig_rows, padding: padding + orig_cols]
        result_con = result_con[padding:padding + orig_rows, padding: padding + orig_cols]
        
        _debug_output(sample_info, result_seg, result_con)


def main(_):
    if not FLAGS.trained_checkpoint:
        raise ValueError('A trained checkpoint must be set for evaluation')

    tf.logging.set_verbosity(tf.logging.INFO)

    evaluate(FLAGS.trained_checkpoint, FLAGS.pixel_threshold, FLAGS.contour_threshold)


tf.app.flags.DEFINE_string(
    'trained_checkpoint', None,
    'The checkpoint to initialize evaluation from')

tf.app.flags.DEFINE_float(
    'pixel_threshold', 0.5,
    'The threshold above which a prediction is considered a positive pixel')

tf.app.flags.DEFINE_float(
    'contour_threshold', 0.5,
    'The threshold below which a prediction not considered a boundary')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()
