import tensorflow as tf
import train
import data
import model
import util
import numpy as np
import scipy
from collections import OrderedDict
from scipy import ndimage as ndi
from skimage import morphology
from skimage import filters
from matplotlib import pyplot as plt

OVERLAP_CONST = 2

_DEBUG_ = True
_DEBUG_WRITE_ = False


# Run-length encoding taken from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1
        prev = b

    return run_lengths


def prob_to_rles(labels):
    for i in range(1, labels.max() + 1):
        yield rle_encoding(labels == i)


# -------------------------------------------------
# Image transforms
def threshold(img, method='otsu'):
    """
        Return binary image based on threshold method
    """
#     if _DEBUG_:
#         from skimage.filters import try_all_threshold
#         fig, ax = try_all_threshold(img, figsize=(10, 8), verbose=False)
#         plt.show()
#     
    if method == 'otsu':
        thresh = filters.threshold_otsu(img)
    elif method == 'mean':
        thresh = filters.threshold_mean(img)
    else:
        thresh = filters.threshold_li(img)
        
    binary = img > thresh
    return binary


def close_filter(img, rad=2, times=1):
    selem = morphology.disk(rad)
    close_img = img
    for i in range(times):
        close_img = morphology.closing(close_img, selem)
        
    return close_img


def open_filter(img, rad=2, times=1):
    selem = morphology.disk(rad)
    open_img = img
    for i in range(times):
        open_img = morphology.opening(open_img, selem)
        
    return open_img


def run_transforms(imga, transforms):

    transformed_image = np.copy(imga)
    for method, kwargs in transforms.items():
        imgt = method(transformed_image, **kwargs)
        if _DEBUG_:
            util.plot_compare(transformed_image, imgt, title1=f"pre-{method.__name__}", title2=f"post-{method.__name__}")
        transformed_image = imgt
    
    return transformed_image
# -------------------------------------------------    
    
    
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


def post_process(result_seg, result_con):
    
    transforms_seg = OrderedDict()
    transforms_seg[threshold] = {}

    transforms_con = OrderedDict()
    transforms_con[threshold] = {}
    #transforms_con[open_filter] = {}
    
    thresh_seg = run_transforms(result_seg, transforms_seg)
    #thresh_con = run_transforms(result_con, transforms_con)
    
#     labels_con = morphology.label(thresh_con)
#     if _DEBUG_:
#         util.plot_compare(thresh_con, labels_con, "thresh_con", "labels_con")
    
    #segments = np.logical_and(thresh_seg, np.logical_not(thresh_con))
    segments = (result_seg - 2 * result_con) > 0.05
    if _DEBUG_:
        util.plot_compare(result_seg, result_con, "result_seg", "result_con")
        util.plot_compare(result_seg, segments, "result_seg", "segments")

    labels = morphology.label(segments)
    if _DEBUG_:
        util.plot_compare(segments, labels, "segments", "labels")
    
    distance_seg = ndi.distance_transform_edt(thresh_seg)
    
    result = morphology.watershed(-distance_seg, labels, mask=thresh_seg)
    if _DEBUG_:
        util.plot_compare(labels, result, "labels", "result")

    # TODO: close the result? May help if holes are common but might not if it closes valid cracks
    return result
    

def evaluate(trained_checkpoint, src='test', use_spline=True):
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

        # util._debug_output(data_processor, sample_info['id'], result_seg, result_con, divisors)

        return post_process(result_seg, result_con)


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

        post_process(result_seg, result_con)
    
        #util._debug_output(data_processor, sample_info['id'], result_seg, result_con, divisors)


def main(_):
    if not FLAGS.trained_checkpoint:
        raise ValueError('A trained checkpoint must be set for evaluation')

    tf.logging.set_verbosity(tf.logging.INFO)

    evaluate(FLAGS.trained_checkpoint, FLAGS.src)


tf.app.flags.DEFINE_string(
    'trained_checkpoint', None,
    'The checkpoint to initialize evaluation from')

# tf.app.flags.DEFINE_float(
#     'pixel_threshold', 0.5,
#     'The threshold above which a prediction is considered a positive pixel')

tf.app.flags.DEFINE_string(
    'src', 'test',
    'The source directory to pull data from')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()
