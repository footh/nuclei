import tensorflow as tf
import train
import data
import model
import util
import numpy as np
import scipy
from collections import OrderedDict
from scipy import ndimage as ndi
import skimage
from skimage import morphology
from skimage import filters
from skimage import feature
import datetime
import os
import csv
import sys

# For debugging
from PIL import Image
from skimage.color import label2rgb
from matplotlib import pyplot as plt
from skimage import img_as_ubyte


OVERLAP_CONST = 2

# SIZE_MININUMS = {
#         100: 10,
#         200: 20,
#         300: 25,
#         400: 30,
#         500: 35,
#         600: 40,
#         800: 45,
#         10000: 50
#     }

SIZE_MININUMS = {
        100: 10,
        200: 20,
        300: 30,
        400: 40,
        500: 50,
        600: 60,
        800: 70,
        10000: 80
    }

SIZE_MAX_MEAN_MULT = 6

CON_MULT = 1.8
SEG_THRESH = 0.4

_DEBUG_ = False
_DEBUG_WRITE_ = False


def SET_DEBUG(val=False, write=False, path='eval'):

    if write:
        tf.gfile.MakeDirs(os.path.join('/tmp', 'nuclei', path))

    sys.modules[__name__]._DEBUG_ = val
    sys.modules[__name__]._DEBUG_WRITE_ = write

    tf_setting = tf.logging.DEBUG if val else tf.logging.INFO
    tf.logging.set_verbosity(tf_setting)


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


def rle_labels(labels):
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
    selem = morphology.square(rad)
    close_img = img
    for i in range(times):
        close_img = morphology.closing(close_img, selem)
        
    return close_img


def open_filter(img, rad=2, times=1):
    selem = morphology.square(rad)
    open_img = img
    for i in range(times):
        open_img = morphology.opening(open_img, selem)
        
    return open_img


def remove_small_objects(img, min_size=20):
    return morphology.remove_small_objects(img, min_size=min_size)


def remove_small_holes(img, min_size=20):
    return morphology.remove_small_holes(img, min_size=min_size)


def frangi_filter(img, scale_range=(0, 5), scale_step=0.25):
    return filters.frangi(img, scale_range=scale_range, scale_step=scale_step)


def hessian_filter(img):
    return filters.hessian(img)


def invert(img):
    return skimage.util.invert(img)


def erode(img, rad=2):
    selem = morphology.square(rad)
    return morphology.erosion(img, selem)


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


def size_boundaries(sizes):

    sizes.sort()
    mean = np.mean(sizes)
    std = np.std(sizes)
#     if len(sizes) > 4:
#         mean = np.mean(sizes[2:-2])
#         std = np.std(sizes[2:-2])
#     else:
#         mean = np.mean(sizes)
#         std = np.std(sizes)
        
    tf.logging.debug(f"size mean, std used in size_boundaries: {mean}, {std}")

    for m, size in SIZE_MININUMS.items():
        if mean < m:
            size_min = size
            break

    size_max = int(SIZE_MAX_MEAN_MULT * mean)
    
    return size_min, size_max


def split_labels(labels):
    result = np.zeros(labels.shape)
    cur_label = 1
    for i in range(1, labels.max() + 1):
        mask = (labels == i)
        size = np.sum(mask)
        print(f"size: {size}")
        distance = ndi.distance_transform_edt(mask)
        util.plot_compare(mask, distance, "mask", "distance")
        #local_max = feature.peak_local_max(distance, indices=False, num_peaks=2, labels=mask)
        lmc = feature.peak_local_max(distance, num_peaks=2, footprint=np.ones((15, 15)), labels=mask)
        print(f"lmc: {lmc}")
        local_max = np.zeros(labels.shape)
        local_max[lmc[:,0], lmc[:,1]] = 1
        print(f"local_max TOTAL: {np.sum(local_max)}")
        label_max = morphology.label(local_max)
        label_max_filled = morphology.watershed(-distance, label_max, mask=mask)
        util.plot_compare(label_max, label_max_filled, "label_max", "label_max_filled")
        for j in range(1, label_max_filled.max() + 1):
            result[label_max_filled == j] = cur_label
            cur_label += 1
            
    return result


def post_process(result_seg, result_con, sample_id=None):
    
    transforms_seg = OrderedDict()
    transforms_seg[threshold] = {}
    #transforms_seg[remove_small_objects] = {}
    #transforms_seg[remove_small_holes] = {}

    transforms_con = OrderedDict()
    transforms_con[invert] = {}
    transforms_con[frangi_filter] = {}
    # transforms_con[threshold] = {'method': 'li'}
    
    thresh_seg = run_transforms(result_seg, transforms_seg)
    
    #np.save('/tmp/nuclei/con.npy', result_con)
    trans_con = run_transforms(result_con, transforms_con)
    trans_con = trans_con / np.max(trans_con)
    trans_con = scipy.stats.gmean(np.dstack((result_con, trans_con)), axis=2)
    #np.save('/tmp/nuclei/con2.npy', trans_con)
    #np.save('/tmp/nuclei/res2.npy', result_con)
    #return
    if _DEBUG_:
        util.plot_compare(result_con, trans_con, "result_con", "trans_con")
    
    #segments = np.logical_and(thresh_seg, np.logical_not(trans_con))
    #segments = (result_seg - CON_MULT * result_con) > SEG_THRESH
    segments = (result_seg - CON_MULT * trans_con) > SEG_THRESH
    if _DEBUG_:
        util.plot_compare(result_seg, result_con, "result_seg", "result_con")
        util.plot_compare(result_seg, segments, "result_seg", "segments")
        
    #segments_cl = close_filter(segments)
    segments_cl = segments
    if _DEBUG_:
        util.plot_compare(segments, segments_cl, "segments", "segments_cl")

    labels = morphology.label(segments_cl)
    if _DEBUG_:
        util.plot_compare(segments_cl, label2rgb(labels, bg_label=0), "segments", "labels")
    
    distance_seg = ndi.distance_transform_edt(thresh_seg)
    
    result = morphology.watershed(-distance_seg, labels, mask=thresh_seg)
    if _DEBUG_:
        util.plot_compare(label2rgb(labels, bg_label=0), label2rgb(result, bg_label=0), "labels", "result")
    tf.logging.info(f"Result label count: {result.max()}")

    sizes = []
    for i in range(1, result.max() + 1):
        sizes.append(np.sum(result == i))
        
    size_min, size_max = size_boundaries(sizes)
    tf.logging.info(f"Size boundaries (min, max): {size_min}, {size_max}")
    
    result_sized = np.zeros(result.shape, dtype=result.dtype)
    size_misses = 0
    for i in range(1, result.max() + 1):
        size = np.sum(result == i)
        if size > size_min and size < size_max:
            result_sized[result==i] = i - size_misses
        else:
            size_misses += 1
    
    if _DEBUG_:
        util.plot_compare(label2rgb(result, bg_label=0), label2rgb(result_sized, bg_label=0), "result", "result_sized")
        #util.plot_hist(sizes)
    tf.logging.info(f"Shape: {result.shape}, Size data, min: {min(sizes)}, max: {max(sizes)}, avg: {np.mean(sizes):.4f}, std: {np.std(sizes):.4f}")
    tf.logging.info(f"Result label count (small labels removed): {result_sized.max()}")

#     result_sized_split = split_labels(result_sized)
#     if _DEBUG_:
#         util.plot_compare(label2rgb(result_sized, bg_label=0), label2rgb(result_sized_split, bg_label=0), "result_sized", "result_sized_split")
    
    if _DEBUG_WRITE_:
        if sample_id is None: sample_id = 'result_sized'
        d_img  = img_as_ubyte(label2rgb(result_sized, bg_label=0))
        Image.fromarray(d_img).save(f"/tmp/nuclei/{FLAGS.debug_path}/{sample_id}.png")
    
    # **DONE** TODO: close the result? May help if holes are common but might not if it closes valid cracks
    # **DONE** worked TODO: may also consider a close on 'segments'. 1-pixel borders may not be valid divisions (contours tend to create much bigger separations)
    # **DONE** made worse TODO: may also consider an open on 'segments'. I've seen some very thin connections that were invalid
    # **DONE** fixing spline helped TODO: remove really small labels, on purple ones, there's a lot of salt that gets labelled and can cause a segment INSIDE a larger one
    # **DONE** fixing spline helped Definitely should consider running an open to remove salt especially on purple ones see #0-8
    # **DONE** no vas TODO: another fix for above is to do the watershed separation method (see valid #52)
    # TODO: see clear_border method which removes dots near borders
    return result_sized


def evaluate(trained_checkpoint, src='test', use_spline=True):
    if FLAGS.debug_path is not None:
        SET_DEBUG(True, True, FLAGS.debug_path)
    
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

    rle_results = []
    #for cnt in range(0, 1):
    for cnt in range(0, data_processor.mode_size(mode='test')):
        sample_tiles, sample_info = data_processor.batch_test(offset=cnt, overlap_const=OVERLAP_CONST)
        if _DEBUG_WRITE_:
            data_processor.copy_id(sample_info['id'], src='debug')
        tf.logging.info(f"Evaluating file {cnt}, id: {sample_info['id']}")
    
        # Prediction --------------------------------
        tile_rows, tile_cols = sample_tiles.shape[0:2]
        tf.logging.info(f"tile_rows, tile_cols: {tile_rows}, {tile_cols}")
        
        sample_batch = sample_tiles.reshape(tile_rows * tile_cols, *sample_tiles.shape[2:])

        # TODO: parameterize
        batch_size = 8
        pred_batches = []
        for i in range(0, sample_batch.shape[0], batch_size):
            pred_batch = sess.run(pred_full, feed_dict={img_input: sample_batch[i:i + batch_size]})
            pred_batches.append(pred_batch)

        sample_pred = np.concatenate(pred_batches)
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
        
        pad_row = sample_info['pad_row']
        pad_col = sample_info['pad_col']
        #orig_rows, orig_cols = sample_info['orig_shape'][0:2]
        tf.logging.info(f"original shape: {sample_info['orig_shape']}")
        result_seg = result_seg[pad_row[0]:-pad_row[1], pad_col[0]:-pad_col[1]]
        result_con = result_con[pad_row[0]:-pad_row[1], pad_col[0]:-pad_col[1]]
        tf.logging.info(f"prediction final shape: {result_seg.shape}")

        # util._debug_output(data_processor, sample_info['id'], result_seg, result_con, divisors)

        result = post_process(result_seg, result_con, sample_id=sample_info['id'])
        
        for rle_label in rle_labels(result):
            rle_results.append([sample_info['id']] + [rle_label])
            
    return rle_results


def evaluate_abut(trained_checkpoint, src='test', pixel_threshold=0.5, contour_threshold=0.5):
    # TODO: parameterize
    window_size = train.IMG_SIZE
    
    sess = tf.InteractiveSession()

    with tf.variable_scope(f"{train.MODEL_SCOPE}/data"):
        img_input = tf.placeholder(tf.float32, [None, window_size, window_size, 3], name='img_input')

    pred_full = build_model(img_input)

    train.restore_from_checkpoint(trained_checkpoint, sess)

    data_processor = data.DataProcessor(src=src, img_size=window_size, testing_pct=100)

    for cnt in range(5, 10):
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

        # util._debug_output(data_processor, sample_info['id'], result_seg, result_con, divisors)

        result = post_process(result_seg, result_con)
        
        for rle_label in rle_labels(result):
            rle_results.append([sample_info['id']] + [rle_label])

    return rle_results


def main(_):
    if not FLAGS.trained_checkpoint:
        raise ValueError('A trained checkpoint must be set for evaluation')

    tf.logging.set_verbosity(tf.logging.INFO)

    rle_results = evaluate(FLAGS.trained_checkpoint, FLAGS.src)
    
    if FLAGS.submission_file is not None:
        submission_file_name = f"submission-{FLAGS.submission_file}-{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
        submission_file_name = os.path.join('submissions', submission_file_name)
        with open(submission_file_name, 'w') as submission_file:
            wr = csv.writer(submission_file, delimiter=',')
            wr.writerow(['ImageId', 'EncodedPixels'])
    
            for rle_result in rle_results:
                enc = ""
                for r in rle_result[1]:
                    enc += f"{r} "
                
                wr.writerow([rle_result[0], enc])


tf.app.flags.DEFINE_string(
    'trained_checkpoint', None,
    'The checkpoint to initialize evaluation from')

tf.app.flags.DEFINE_string(
    'src', 'test',
    'The source directory to pull data from')

tf.app.flags.DEFINE_string(
    'submission_file', None,
    'The name of the submission file to save (date will be automatically appended)')

tf.app.flags.DEFINE_string(
    'debug_path', None,
    'Both turns on debugging and indicates the sub-directory under /tmp/nuclei to save debug files')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()
