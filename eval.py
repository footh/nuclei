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
from scipy.spatial import distance
import cv2
import datetime
import os
import csv
import sys

# For debugging
from PIL import Image
from skimage.color import label2rgb
from skimage import img_as_ubyte
from imgaug import augmenters as iaa


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

# newsz2 yields 0.448 on LB
# SIZE_MININUMS = {
#         100: 10,
#         200: 20,
#         300: 30,
#         400: 40,
#         500: 50,
#         600: 60,
#         800: 70,
#         10000: 80
#     }

# newsz3 yields 0.450 on LB
SIZE_MININUMS = {
     
        100: 15,
        200: 25,
        300: 35,
        400: 45,
        500: 55,
        600: 65,
        800: 75,
        10000: 85
    }

# newsz4 yields 0.447 on LB
# SIZE_MININUMS = {
#     
#         100: 20,
#         200: 30,
#         300: 40,
#         400: 50,
#         500: 60,
#         600: 70,
#         800: 80,
#         10000: 90
#     }

# IMAGE_AUGS = [
#     iaa.AdditiveGaussianNoise(scale=0.025 * 255),
#     iaa.AdditiveGaussianNoise(scale=0.05 * 255),
#     iaa.GaussianBlur(sigma=0.5),
#     iaa.GaussianBlur(sigma=1.0),
#     iaa.ContrastNormalization(0.9),
#     iaa.ContrastNormalization(1.1),
#     iaa.Multiply(0.8),
#     iaa.Multiply(1.2)
# ]

IMAGE_AUGS = [
    iaa.AdditiveGaussianNoise(scale=0.03 * 255),
    iaa.GaussianBlur(sigma=0.67),
    iaa.ContrastNormalization(1.1),
    iaa.Multiply(1.1)
]

SIZE_MAX_MEAN_MULT = 6

CON_MULT = 2.0
SEG_THRESH = 0.5

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


def rle_labels(labels, dilate=None):
    for i in range(1, labels.max() + 1):
        x = (labels == i)
        if dilate is not None:
            selem = morphology.disk(dilate)
            x = morphology.binary_dilation(x, selem=selem)
        
        yield rle_encoding(x)


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


def affine_augment(tiles, flip, rotation):
    """
        Run simple flip and rotation augmentation on batch of inputs
    """
    result = np.zeros(tiles.shape)
    for i, tile in enumerate(tiles):
        if flip:
            result[i] = tile[:, ::-1]

        result[i] = np.rot90(tile, rotation)

    return result

    
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

    if len(sizes) == 0:
        return 0, 0

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
    rprops = skimage.measure.regionprops(labels)

    result = np.zeros(labels.shape)
    cur_label = 1
    for i in range(1, labels.max() + 1):
        if rprops[i - 1].convex_area / rprops[i - 1].filled_area > 1.1:
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


CONVEX_OVERFLOW_MIN = 1.1  # Convex hull area overflow ratio
CONVEX_DIST_MIN = 0.1  # Far point distance must be this (percentage) amount of the minor axis length
DIST_BASE_RATIO_MIN = 0.20  # Far point distance must be this (percentage) amount of the defect base


def split_convexity(labels):
    rprops = skimage.measure.regionprops(labels)
    selem = skimage.morphology.disk(1)

    result = np.zeros(labels.shape, dtype=np.int16)
    cur_label = 1
    for i in range(1, labels.max() + 1):
        mask = (labels == i).astype(np.uint8)
        mask = morphology.closing(mask, selem)
        if rprops[i - 1].convex_area / rprops[i - 1].filled_area > CONVEX_OVERFLOW_MIN:
            minor_axis_len = rprops[i - 1].minor_axis_length

            _, contours, _ = cv2.findContours(mask, 2, 1)
            cnt = contours[0]
            hull = cv2.convexHull(cnt, returnPoints=False)

            defects = cv2.convexityDefects(cnt, hull)

            key_points = []
            starts = []
            ends = []
            for j in range(defects.shape[0]):
                s, e, f, d = defects[j, 0]
                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                norm_dist = (d / 256.)

                # Far point distance as a ratio of minor axis length
                norm_dist_ratio = norm_dist / minor_axis_len
                # Ratio of far point distance to base
                dist_base_ratio = norm_dist / distance.euclidean(start, end)

                if norm_dist_ratio > CONVEX_DIST_MIN and dist_base_ratio > DIST_BASE_RATIO_MIN:
                    starts.append(start)
                    ends.append(end)
                    key_points.append(far)

            if len(key_points) == 1:
                mid = (int((starts[0][0] + ends[0][0]) / 2), int((starts[0][1] + ends[0][1]) / 2))
                util.draw_long_line(mask, mid, key_points[0])
            # elif len(key_points) >= 2:
            #     for k in range(len(key_points)):
            #         p1 = key_points[k]
            #
            #         nearest = None
            #         min_dist = np.inf
            #         for l in range(len(key_points)):
            #             if k != l:
            #                 p2 = key_points[l]
            #                 dist = distance.euclidean(p1, p2)
            #                 if dist < min_dist:
            #                     min_dist = dist
            #                     nearest = p2
            #
            #         if nearest is not None:
            #             cv2.line(mask, p1, nearest, [0, 0, 0], 2)

        labels_new = morphology.label(mask)
        for j in range(1, labels_new.max() + 1):
            result[labels_new == j] = cur_label
            cur_label += 1

    return result


def super_frangi(img):
    inv_img = invert(img)

    frangis = []

    frangis.append(frangi_filter(inv_img, scale_range=(0, 8), scale_step=0.4))
    frangis.append(frangi_filter(inv_img, scale_range=(0, 4), scale_step=0.1))
    frangis.append(frangi_filter(inv_img, scale_range=(1, 10), scale_step=0.5))
    frangis.append(frangi_filter(inv_img, scale_range=(1, 15), scale_step=1))

    frangis = [(f / np.max(f)) for f in frangis]

    return scipy.stats.gmean(np.dstack(frangis), axis=2)
    # return np.mean(np.dstack(frangis), axis=2)


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

    # sf = super_frangi(result_con)
    # if _DEBUG_:
    #     util.plot_compare(trans_con, sf, "frangi", "super frangi")

    trans_con = scipy.stats.gmean(np.dstack((result_con, trans_con)), axis=2)
    # trans_con = scipy.stats.gmean(np.dstack((result_con, sf)), axis=2)
    # np.save('/tmp/nuclei/con2.npy', trans_con)
    # np.save('/tmp/nuclei/res2.npy', result_con)
    # return
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
    # if _DEBUG_:
    #     util.plot_compare(segments, segments_cl, "segments", "segments_cl")

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
    if len(sizes) > 0:
        tf.logging.info(f"Shape: {result.shape}, Size data, min: {min(sizes)}, max: {max(sizes)}, avg: {np.mean(sizes):.4f}, std: {np.std(sizes):.4f}")
    tf.logging.info(f"Result label count (small labels removed): {result_sized.max()}")

    if _DEBUG_WRITE_:
        if sample_id is None: sample_id = 'result_sized'
        d_img = img_as_ubyte(label2rgb(result_sized, bg_label=0))
        Image.fromarray(d_img).save(f"/tmp/nuclei/{FLAGS.debug_path}/{sample_id}.png")
        np.save(f"/tmp/nuclei/{FLAGS.debug_path}/{sample_id}.npy", result_sized)

    # **DONE** made slightly worse TODO: close the result? May help if holes are common but might not if it closes valid cracks
    # **DONE** worked TODO: may also consider a close on 'segments'. 1-pixel borders may not be valid divisions (contours tend to create much bigger separations)
    # **DONE** made worse TODO: may also consider an open on 'segments'. I've seen some very thin connections that were invalid
    # **DONE** fixing spline helped Definitely should consider running an open to remove salt especially on purple ones see #0-8
    # **DONE** no vas TODO: another fix for above is to do the watershed separation method (see valid #52)
    # ** DONE **, tried dilation on labels, bad
    # TODO: see clear_border method which removes dots near borders

    # Morphology split
    # result_sized_split = split_labels(result_sized)
    # if _DEBUG_:
    #     util.plot_compare(label2rgb(result_sized, bg_label=0), label2rgb(result_sized_split, bg_label=0), "result_sized", "result_sized_split")

    # Convexity split
    # result_sized_sc = split_convexity(result_sized)
    # result_sized_split = np.zeros(result_sized_sc.shape, dtype=result.dtype)
    # size_misses = 0
    # for i in range(1, result_sized_sc.max() + 1):
    #     size = np.sum(result_sized_sc == i)
    #     if size > size_min and size < size_max:
    #         result_sized_split[result_sized_sc == i] = i - size_misses
    #     else:
    #         size_misses += 1
    #
    # if _DEBUG_:
    #     util.plot_compare(label2rgb(result_sized, bg_label=0), label2rgb(result_sized_split, bg_label=0), "result_sized", "result_sized_split")
    # if _DEBUG_WRITE_:
    #     d_img = img_as_ubyte(label2rgb(result_sized_split, bg_label=0))
    #     Image.fromarray(d_img).save(f"/tmp/nuclei/{FLAGS.debug_path}/{sample_id}-split.png")
    #     np.save(f"/tmp/nuclei/{FLAGS.debug_path}/{sample_id}-split.npy", result_sized_split)

    # Dilation
    # result_sized_d = np.zeros(result.shape, dtype=result.dtype)
    # selem = morphology.square(2)
    # for i in range(1, result_sized.max() + 1):
    #     mask = (result_sized == i)
    #     mask = morphology.binary_dilation(mask, selem)
    #     result_sized_d[mask] = i
    #
    # if _DEBUG_:
    #     util.plot_compare(label2rgb(result_sized, bg_label=0), label2rgb(result_sized_d, bg_label=0), 'result_sized', 'result_sized_d')

    # Closing
    # result_sized_c = np.zeros(result.shape, dtype=result.dtype)
    # selem = morphology.square(2)
    # for i in range(1, result_sized.max() + 1):
    #     mask = (result_sized == i)
    #     mask = morphology.binary_closing(mask, selem)
    #     result_sized_c[mask] = i
    #
    # if _DEBUG_:
    #     util.plot_compare(label2rgb(result_sized, bg_label=0), label2rgb(result_sized_c, bg_label=0), 'result_sized', 'result_sized_c')

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

    # train.restore_from_checkpoint(trained_checkpoint, sess)
    checkpoints = [ckpt.strip() for ckpt in trained_checkpoint.split(',')]
    tf.logging.info(f"Checkpoints: {checkpoints}")

    data_processor = data.DataProcessor(src=src, img_size=window_size, testing_pct=100)
    
    if use_spline:
        spline_window = _spline_window(window_size)

    rle_results = []
    for cnt in range(0, data_processor.mode_size(mode='test')):

        all_predictions = []
        for ckpt in checkpoints:
            ckpt_file, inv = ckpt.split('*')
            train.restore_from_checkpoint(ckpt_file, sess)

            sample_tiles, sample_info = data_processor.batch_test(offset=cnt, overlap_const=OVERLAP_CONST, invert=int(inv))
            if _DEBUG_WRITE_:
                data_processor.copy_id(sample_info['id'], src=FLAGS.debug_path)
            tf.logging.info(f"Evaluating file {cnt}, id: {sample_info['id']}")

            # Prediction --------------------------------
            tile_rows, tile_cols = sample_tiles.shape[0:2]
            tf.logging.info(f"tile_rows, tile_cols: {tile_rows}, {tile_cols}")

            sample_batch = sample_tiles.reshape(tile_rows * tile_cols, *sample_tiles.shape[2:])

            # Run predictions on each affine augmentation (rotation, flip)
            for flip in range(1):
                for rotation in range(4):
                    tf.logging.info(f"flip, rotation: {flip}, {rotation}")
                    sample_batch_aug = affine_augment(sample_batch, flip, rotation)

                    # TODO: parameterize
                    batch_size = 8
                    pred_batches = []
                    for i in range(0, sample_batch_aug.shape[0], batch_size):
                        pred_batch = sess.run(pred_full, feed_dict={img_input: sample_batch_aug[i:i + batch_size]})
                        pred_batches.append(pred_batch)

                    sample_pred = np.concatenate(pred_batches)
                    tf.logging.info(f"sample_pred.shape (post-batch): {sample_pred.shape}")

                    # Reverse the augmentation and store in list
                    sample_pred = affine_augment(sample_pred, flip, -rotation)

                    all_predictions.append(sample_pred)

            # sample_batch_raw = data_processor.preprocess(sample_batch, reverse=True)
            # for aug in IMAGE_AUGS:
            #     sample_batch_aug = aug.augment_images(sample_batch_raw)
            #     sample_batch_aug = data_processor.preprocess(sample_batch_aug)
            #
            #     # TODO: parameterize
            #     batch_size = 8
            #     pred_batches = []
            #     for i in range(0, sample_batch_aug.shape[0], batch_size):
            #         pred_batch = sess.run(pred_full, feed_dict={img_input: sample_batch_aug[i:i + batch_size]})
            #         pred_batches.append(pred_batch)
            #
            #     sample_pred = np.concatenate(pred_batches)
            #     tf.logging.info(f"sample_pred.shape (post-batch): {sample_pred.shape}")
            #
            #     all_predictions.append(sample_pred)

        # Take the mean of the predictions
        sample_pred = np.mean(all_predictions, axis=0)
        tf.logging.info(f"sample_pred.shape (post-mean): {sample_pred.shape}")

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
        
        # Trying dilation (and closing) here resulted in duplicate pixels which the submission code caught
        for rle_label in rle_labels(result, dilate=None):
            rle_results.append([sample_info['id']] + [rle_label])
            
    return rle_results


def test():
    data_processor = data.DataProcessor(src='vtest', img_size=256, testing_pct=100)
    sample_tiles, sample_info = data_processor.batch_test(offset=0, overlap_const=2)
    tile_rows, tile_cols = sample_tiles.shape[0:2]
    sample_batch = sample_tiles.reshape(tile_rows * tile_cols, *sample_tiles.shape[2:])

    sample_batch_raw = data_processor.preprocess(sample_batch, reverse=True)

    for i, tile in enumerate(sample_batch_raw):
        print(f"OG {i}")
        Image.fromarray(np.asarray(tile, dtype=np.uint8)).save(f"/tmp/nuclei/data/tile-{i}-orig.png")

    for a, aug in enumerate(IMAGE_AUGS):
        sample_batch_aug = aug.augment_images(sample_batch_raw)

        for i, tile in enumerate(sample_batch_aug):
            print(f"max, min: {np.max(tile)}, {np.min(tile)}")
            Image.fromarray(np.asarray(tile, dtype=np.uint8)).save(f"/tmp/nuclei/data/tile-{i}-aug{a}-{aug.name}.png")

    sample_batch = data_processor.preprocess(sample_batch_aug)
    for tile in sample_batch:
        print(f"max, min: {np.max(tile)}, {np.min(tile)}")


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
    'Comma separated list of checkpoints. Checkpoint path is followed by * and 0 or 1 to indicate image inversion')

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
