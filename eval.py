import tensorflow as tf
import train
import data
import model
import numpy as np
import shutil
import os
from PIL import Image
from skimage import filters
from skimage import morphology
import scipy
from scipy.interpolate.interpolate import spline

OVERLAP_CONST = 2

def _debug_output(dp, sample_id, result_seg, result_con, divisors):
    print(f"seg: {np.sum(result_seg > 1)}")
    print(f"con: {np.sum(result_con > 1)}")

    seg_name = f"{sample_id}-{data.IMG_SEGMENT}.type.{data.IMG_EXT}"
    con_name = f"{sample_id}-{data.IMG_CONTOUR}.type.{data.IMG_EXT}"
    
    dp.copy_id(sample_id, src='debug')

    result_seg_out = np.asarray(result_seg * 255, dtype=np.uint8)
    result_con_out = np.asarray(result_con * 255, dtype=np.uint8)
    Image.fromarray(result_seg_out).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'pred')))
    Image.fromarray(result_con_out).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'pred')))
    
    selem = morphology.disk(3)

    # Pipeline #1: Preds -> Close -> Thresh -> Diff
    result_seg_c = morphology.closing(result_seg, selem)
    result_con_c = morphology.closing(result_con, selem)

    result_seg_c = np.asarray(result_seg_c * 255, dtype=np.uint8)
    result_con_c = np.asarray(result_con_c * 255, dtype=np.uint8)
    Image.fromarray(result_seg_c).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'predc1')))
    Image.fromarray(result_con_c).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'predc1')))

    prob_seg_c = filters.threshold_otsu(result_seg_c)
    prob_con_c = filters.threshold_otsu(result_con_c)
    print(f"thresh_seg, thresh_con: {prob_seg_c}, {prob_con_c}")

    thresh_seg_c1 = np.asarray(result_seg_c > prob_seg_c, dtype=np.uint8) * 255
    thresh_con_c1 = np.asarray(result_con_c > prob_con_c, dtype=np.uint8) * 255
    Image.fromarray(thresh_seg_c1).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'thr1')))
    Image.fromarray(thresh_con_c1).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thr1')))

    thresh_con_c1e = morphology.erosion(thresh_con_c1, morphology.square(2))
    Image.fromarray(thresh_con_c1e).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thr1e')))

    diff_1 = np.maximum(thresh_seg_c1 - thresh_con_c1e, 0)
    Image.fromarray(diff_1).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('con.type', 'diff1')))

    # Pipeline #2: Preds -> Thresh -> Close -> Diff
    prob_seg = filters.threshold_otsu(result_seg)
    prob_con = filters.threshold_otsu(result_con)
    print(f"thresh_seg, thresh_con: {prob_seg}, {prob_con}")

    thresh_seg = np.asarray(result_seg > prob_seg, dtype=np.uint8) * 255
    thresh_con = np.asarray(result_con > prob_con, dtype=np.uint8) * 255
    Image.fromarray(thresh_seg).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'thr2')))
    Image.fromarray(thresh_con).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thr2')))

    thresh_seg_c2 = morphology.closing(thresh_seg, selem)
    thresh_con_c2 = morphology.closing(thresh_con, selem)
    Image.fromarray(thresh_seg_c2).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'thrc2')))
    Image.fromarray(thresh_con_c2).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thrc2')))
    diff_2 = np.maximum(thresh_seg_c2 - thresh_con_c2, 0)
    Image.fromarray(diff_2).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('con.type', 'diff2')))

    #m = np.max(divisors)
    #divisors = np.asarray((divisors / m) * 255, dtype=np.uint8)
    #Image.fromarray(divisors).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('con.type', 'div')))

    
def _spline_window(window_size, power=2):
    """
        Squared spline (power=2) window function:
        https://www.wolframalpha.com/input/?i=y%3Dx**2,+y%3D-(x-2)**2+%2B2,+y%3D(x-4)**2,+from+y+%3D+0+to+2
    """
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0

    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0

    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)

    wind = np.expand_dims(wind, 1)
    wind = wind * wind.transpose(1, 0)
    tf.logging.debug(f"spline window shape: {wind.shape}")
    return wind
    

def build_model(img_input):
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
    # -------------------------------------------    

    return pred_full


def evaluate(trained_checkpoint, src='test', pixel_threshold=0.5, contour_threshold=0.5, use_spline=True):
    # TODO: parameterize
    window_size = train.IMG_SIZE
    
    sess = tf.InteractiveSession()

    with tf.variable_scope(f"{train.MODEL_SCOPE}/data"):
        img_input = tf.placeholder(tf.float32, [None, window_size, window_size, 3], name='img_input')

    pred_full = build_model(img_input)

    train.restore_from_checkpoint(trained_checkpoint, sess)

    data_processor = data.DataProcessor(src=src, img_size=window_size, testing_pct=100)
    
    if use_spline:
        spline_window = _spline_window(window_size)

    for cnt in range(10, 11):
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
                if use_spline:
                    seg = seg * spline_window
                    con = con * spline_window
                result_seg[row_start:row_start + window_size, col_start:col_start + window_size] += seg
                result_con[row_start:row_start + window_size, col_start:col_start + window_size] += con
                divisors[row_start:row_start + window_size, col_start:col_start + window_size] += 1.
                
        if use_spline:
            result_seg = result_seg / OVERLAP_CONST ** 2
            result_con = result_con / OVERLAP_CONST ** 2
        else:
            result_seg = result_seg / divisors
            result_con = result_con / divisors
        
        padding = sample_info['padding']
        orig_rows, orig_cols = sample_info['orig_shape'][0:2]
        result_seg = result_seg[padding:padding + orig_rows, padding: padding + orig_cols]
        result_con = result_con[padding:padding + orig_rows, padding: padding + orig_cols]
        
        _debug_output(data_processor, sample_info['id'], result_seg, result_con, divisors)


def evaluate_abut(trained_checkpoint, src='test', pixel_threshold=0.5, contour_threshold=0.5):
    # TODO: parameterize
    window_size = train.IMG_SIZE
    
    sess = tf.InteractiveSession()

    with tf.variable_scope(f"{train.MODEL_SCOPE}/data"):
        img_input = tf.placeholder(tf.float32, [None, window_size, window_size, 3], name='img_input')

    pred_full = build_model(img_input)

    train.restore_from_checkpoint(trained_checkpoint, sess)

    data_processor = data.DataProcessor(src=src, img_size=window_size, testing_pct=100)

    for cnt in range(5,10):
        sample_tiles, sample_info = data_processor.batch_test_abut(offset=cnt)
        
        # Prediction --------------------------------
        tile_rows, tile_cols = sample_tiles.shape[0:2]
        tf.logging.info(f"tile_rows, tile_cols: {tile_rows}, {tile_cols}")
        
        # TODO: with large number of tiles, may need to batch this further
        sample_batch = sample_tiles.reshape(tile_rows * tile_cols, *sample_tiles.shape[2:])
    
        sample_pred = sess.run(pred_full, feed_dict={img_input: sample_batch})
        tf.logging.info(f"sample_pred.shape: {sample_pred.shape}")
    
        sample_pred = sample_pred.reshape(tile_rows, tile_cols, *sample_pred.shape[1:])
        # -------------------------------------------

        full_pred_rows = tile_rows * window_size
        full_pred_cols = tile_cols * window_size
        tf.logging.info(f"full_pred_rows, full_pred_cols: {full_pred_rows}, {full_pred_cols}")
    
        result_seg = np.zeros((full_pred_rows, full_pred_cols), dtype=np.float32)
        result_con = np.zeros((full_pred_rows, full_pred_cols), dtype=np.float32)
        divisors = np.zeros((full_pred_rows, full_pred_cols), dtype=np.float32)
    
        for i, row_start in enumerate(range(0, full_pred_rows, window_size)):
            for j, col_start in enumerate(range(0, full_pred_cols, window_size)):
                seg = sample_pred[i, j, :, :, 0]
                con = sample_pred[i, j, :, :, 1]
                result_seg[row_start:row_start + window_size, col_start:col_start + window_size] += seg
                result_con[row_start:row_start + window_size, col_start:col_start + window_size] += con
                divisors[row_start:row_start + window_size, col_start:col_start + window_size] += 1.
    
        padding_row = sample_info['padding_row']
        padding_col = sample_info['padding_col']
        result_seg = result_seg[padding_row[0]:full_pred_rows - padding_row[1], padding_col[0]:full_pred_cols - padding_col[1]]
        result_con = result_con[padding_row[0]:full_pred_rows - padding_row[1], padding_col[0]:full_pred_cols - padding_col[1]]
    
        _debug_output(data_processor, sample_info['id'], result_seg, result_con, divisors)


def main(_):
    if not FLAGS.trained_checkpoint:
        raise ValueError('A trained checkpoint must be set for evaluation')

    tf.logging.set_verbosity(tf.logging.INFO)

    evaluate(FLAGS.trained_checkpoint, FLAGS.src, FLAGS.pixel_threshold, FLAGS.contour_threshold)


tf.app.flags.DEFINE_string(
    'trained_checkpoint', None,
    'The checkpoint to initialize evaluation from')

tf.app.flags.DEFINE_float(
    'pixel_threshold', 0.5,
    'The threshold above which a prediction is considered a positive pixel')

tf.app.flags.DEFINE_float(
    'contour_threshold', 0.5,
    'The threshold below which a prediction not considered a boundary')

tf.app.flags.DEFINE_string(
    'src', 'test',
    'The source directory to pull data from')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()
