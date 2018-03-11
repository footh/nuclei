from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import sys
import os
import pandas as pd
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import measure
from skimage import morphology
import hashlib
import math
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from imgaug import augmenters as iaa

IMG_EXT = 'png'
IMG_CHANNELS = 3
VGG_RGB_MEANS = [123.68, 116.78, 103.94]
PAD_MODE = 'reflect'

MAX_NUM_PER_CLASS = 2**27 - 1  # ~134M

IMG_SRC = 'src'
IMG_CONTOUR = 'con'
IMG_SEGMENT = 'seg'

# CONTOUR_DILATION = {
#         5: 2,
#         10: 2,
#         17: 3,
#         22: 4,
#         30: 5,
#         42: 6,
#         1000: 7
#     }

CONTOUR_DILATION = {
        5: 2,
        10: 3,
        17: 4,
        22: 5,
        30: 6,
        42: 7,
        1000: 8
    }


CONTOUR_FREQ_RATIO = 0.008 # Ratio of positive contour labels that must be in a sample to be considered a hit
CONTOUR_FREQ = 0.5 # Percentage of a batch that must contain contour label hits
CONTOUR_CONTINUE_MAX = 100 # Amount of times to try a different sample before just moving on with the batch

_DEBUG_ = True
_DEBUG_WRITE_ = False


def SET_DEBUG(val=False, write=False):
    if write:
        tf.gfile.MakeDirs('/tmp/nuclei')

    sys.modules[__name__]._DEBUG_ = val
    sys.modules[__name__]._DEBUG_WRITE_ = write

    tf_setting = tf.logging.DEBUG if val else tf.logging.INFO
    tf.logging.set_verbosity(tf_setting)


def _remove_files(src):
    """
        Remove files from src directory (train, test, etc) and sub-directories
    """
    if os.path.isfile(src):
        os.unlink(src)
    elif os.path.isdir(src):
        # map lazy evaluates so must wrap in list to force evaluation
        list(map(_remove_files, [os.path.join(src, fi) for fi in os.listdir(src)])) 


def file_list(src_dir, fname_wildcard='*'):
    """
        Return a list of files from the src_dir directory
    """
    result = []
    search_path = os.path.join(src_dir, f"{fname_wildcard}.{IMG_EXT}")
    for img_path in gfile.Glob(search_path):
        result.append(img_path)

    return result
    

def raw_file_list(src='train'):
    """
        Return a list of files from the raw-data directory for the given src
    """
    src_dir = os.path.join('raw-data', src, '*', 'images')
    return file_list(src_dir)


def as_images(src='train', size=None):
    """
        Return a list of PIL images from the raw-data directory for the given src. Resized to the size argument if given.
    """
    result = []
    flist = file_list(src)
    for f in flist:
        img = Image.open(f)
        if size is not None:
            img = img.resize(size)
        result.append(img)
    
    return result


def kmeans(img_size=(256, 256), clusters=3):
    """
        Kmeans testing
    """
    print(f"Running kmeans...")
    imgs = as_images(size=img_size)
    
    vec_size = img_size[0]*img_size[1]
    x = np.zeros((len(imgs), vec_size), dtype=np.float32)
    for i, img in enumerate(imgs):
        imga = np.asarray(img)
        imga = imga[:,:,0:3]
        imga = np.mean(imga, axis=2)
        x[i] = imga.reshape(vec_size)
    
    print(f"Data processed...")
     
    km = KMeans(n_clusters=clusters, random_state=0).fit(x)
    for i in range(clusters):
        lbl_count = np.sum(km.labels_ == i)
        print(f"cluster {i}: {lbl_count}")
        print(f"cluster {i} ratio: {lbl_count/len(imgs)}")

    return x, km


def _draw_contours(src, dest):
    """
        Finds contours on src array and writes them to dest array
    """
    contours = measure.find_contours(src, 0.5)  # TODO: investigate this parameter
    # assert(len(contours) == 1)
    rprops = measure.regionprops(src // 255)
    minor_axis = rprops[0].minor_axis_length
    assert(minor_axis > 0)

    result = np.zeros(src.shape, dtype=np.uint8)
    for contour in contours:
        contour = contour.astype(int)
        result[contour[:, 0], contour[:, 1]] = 255

    dilation_val = 1
    for maxis_size, selem_size in CONTOUR_DILATION.items():
        if minor_axis < maxis_size:
            dilation_val = selem_size
            break
    
    se = morphology.square(dilation_val)
    result = morphology.dilation(result, se)

    dest = np.maximum(dest, result)
    return dest


def full_mask(raw_file_path, with_contours=True):
    """
        Given the raw_file_path, get the full ground truth mask for that file combining individual masks. If the
        with_contours value is True (default) a contour image will be created.
    """
    masks = []
    search_path = os.path.join(os.path.dirname(raw_file_path), '..', 'masks', f"*.{IMG_EXT}")
    for img_path in gfile.Glob(search_path):
        masks.append(img_path)
            
    img_shape = Image.open(masks[0]).size
    img_shape = (img_shape[1], img_shape[0])
    
    mask = np.zeros(img_shape, dtype=np.uint8)
    contour = None
    if with_contours:
        contour = np.zeros(img_shape, dtype=np.uint8)
    
    for m in masks:
        mask_img = Image.open(m)
        if mask_img.mode is not 'L': raise Exception(f"Mask image is not L mode: {m}")
        mask_img = np.asarray(mask_img)
        mask = np.maximum(mask, mask_img)
        
        if with_contours:
            contour = _draw_contours(mask_img, contour)
        
    return Image.fromarray(mask), Image.fromarray(contour)


def setup(src='train', with_masks=True, early_stop=None):
    """
        Move raw files with full ground truth segment and contour masks over to the src directory. File names
        are id-{type}.png where type is 'src' for source image, 'seg' for segment mask and 'con' for contour mask
    """
    src_dir = os.path.join('model-data', src)
    print(f"Clearing {src_dir} directory...")
    _remove_files(src_dir)

    flist = raw_file_list(src=src)
    for cnt, f in enumerate(flist):
        name, ext = os.path.basename(f).split('.')
        print(f"{cnt + 1}: Processing {name}...")

        if early_stop is not None and cnt > early_stop:
            break

        if with_masks:
            mask, contour = full_mask(f)
        
        shutil.copy2(f, os.path.join(src_dir, f"{name}-{IMG_SRC}.{ext}"))
        if with_masks:
            mask.save(os.path.join(src_dir, f"{name}-{IMG_SEGMENT}.{ext}"))
            contour.save(os.path.join(src_dir, f"{name}-{IMG_CONTOUR}.{ext}"))


def ratio(src='train'):
    """
        Retrieve the ratios of segment and contour masks
    """
    seg_pixels = 0
    seg_total = 0

    src_dir = os.path.join('model-data', src)
    seg_files = file_list(src_dir, fname_wildcard=f"*-{IMG_SEGMENT}")
    tf.logging.info(f"Segment files found: {len(seg_files)}")
    for seg_file in seg_files:
        seg = np.asarray(Image.open(seg_file))
        seg_total += np.prod(seg.shape)
        seg_pixels += np.sum(seg > 0)

    con_pixels = 0
    con_total = 0

    con_files = file_list(src_dir, fname_wildcard=f"*-{IMG_CONTOUR}")
    tf.logging.info(f"Contour files found: {len(con_files)}")
    for con_file in con_files:
        con = np.asarray(Image.open(con_file))
        con_total += np.prod(con.shape)
        con_pixels += np.sum(con > 0)

    assert(seg_total == con_total)

    seg_ratio = seg_pixels / seg_total
    con_ratio = con_pixels / con_total

    return seg_ratio, con_ratio


def which_set(file_id, validation_pct, testing_pct, radix=MAX_NUM_PER_CLASS):
    """
        Determines which data partition the file should belong to.
        (taken from tensorflow speech audio tutorial
    
        Returns string, one of 'train', 'valid', or 'test'.
    """
    file_id_hashed = hashlib.sha1(compat.as_bytes(file_id)).hexdigest()
    percentage_hash = ((int(file_id_hashed, 16) % (radix + 1)) * (100.0 / radix))
    percentage_hash = int(percentage_hash)
    
    if validation_pct == 100:
        return 'valid'
    
    if testing_pct == 100:
        return 'test'

    if percentage_hash < validation_pct:
        result = 'valid'
    elif percentage_hash < (testing_pct + validation_pct):
        result = 'test'
    else:
        result = 'train'

    return result


def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    print(f"shape: {shape}")

    dx = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape[0:2]) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    result = np.zeros(shape, dtype=image.dtype)
    if len(shape) == 2:
        result[:, :] = map_coordinates(image, indices, order=1).reshape(shape[0:2])
    else:
        for i in range(shape[2]):
            result[:, :, i] = map_coordinates(image[:, :, i], indices, order=1).reshape(shape[0:2])

    return result


def et_test():
    img = np.asarray(Image.open('train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e-src.png'))
    imgs = np.asarray(Image.open('train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e-seg.png'))
    imgs = np.expand_dims(imgs, axis=-1)
    imgc = np.asarray(Image.open('train/00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e-con.png'))
    imgc = np.expand_dims(imgc, axis=-1)
    print(f"{img.shape}, {imgs.shape}, {imgc.shape}")
    
    all_img = np.concatenate([img, imgs, imgc], axis=-1)
    print(f"all_img.shape: {all_img.shape}")
    
    imgt = elastic_transform(all_img, all_img.shape[1]*2, all_img.shape[1]*0.08)

    imgts = np.squeeze(imgt[:, :, 4:5])
    imgtc = np.squeeze(imgt[:, :, 5:6])
    
    Image.fromarray(imgt[:,:,0:4]).save('/tmp/nuclei/test-src.png')
    Image.fromarray(imgts).save('/tmp/nuclei/test-seg.png')
    Image.fromarray(imgtc).save('/tmp/nuclei/test-con.png')
    

class DataProcessor:
    
    def __init__(self, src='train', img_size=256, validation_pct=0, testing_pct=0, valid_same=True):
        """
            Build data processor for yielding train, valid and test data
            
            'valid_same' parameter will assure the sampling points on 'valid' mode will be the same across calls
        """
        self.src_dir = os.path.join('model-data', src)
        self.img_size = img_size
        self.validation_pct = validation_pct
        self.testing_pct = testing_pct
        self.data_index = {'train': [], 'valid': [], 'test': []}

        self._generate_data_index()

        self.valid_seeds = None
        if valid_same:
             self.valid_seeds = np.random.rand(self.mode_size(mode='valid'), 2)

        # Color augmentation
        self._color_aug = iaa.Sometimes(0.5,
                                        iaa.Sequential([
                                            iaa.OneOf([
                                                iaa.Sequential([
                                                    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                                    iaa.WithChannels(0, iaa.Add((0, 100))),
                                                    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                                                iaa.Sequential([
                                                    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                                    iaa.WithChannels(1, iaa.Add((0, 100))),
                                                    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                                                iaa.Sequential([
                                                    iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
                                                    iaa.WithChannels(2, iaa.Add((0, 100))),
                                                    iaa.ChangeColorspace(from_colorspace="HSV", to_colorspace="RGB")]),
                                                iaa.WithChannels(0, iaa.Add((0, 100))),
                                                iaa.WithChannels(1, iaa.Add((0, 100))),
                                                iaa.WithChannels(2, iaa.Add((0, 100)))
                                            ])
                                        ], random_order=True))

    def _generate_data_index(self):
        """
            Build the index of files, bucketed by source group
        """
        class_dict = self._classes()

        for class_key, id_list in class_dict.items():
            tf.logging.info(f"Allotting for class: {class_key}")
            for id in id_list:
                idx = which_set(id, self.validation_pct, self.testing_pct, radix=len(id_list))
                self.data_index[idx].append(id)
            
        tf.logging.info(f"Training total: {len(self.data_index['train'])}")
        tf.logging.info(f"Validation total: {len(self.data_index['valid'])}")
        tf.logging.info(f"Testing total: {len(self.data_index['test'])}")

    def _classes(self, file='classes.csv'):
        """
            Returns a dict of class type to file id for the given src
        """
        df = pd.read_csv(file)
    
        # Restrict to files in the processed directory
        all_files = file_list(self.src_dir)
        flist = list({f"{os.path.basename(file).split('-')[0]}.{IMG_EXT}" for file in all_files})        
        df = df.query('filename in @flist')
    
        result = dict(df.groupby(df.foreground + '-' + df.background)['filename'].apply(list))
    
        ttl = 0
        for key, value in result.items():
            id_list = [os.path.splitext(v)[0] for v in value]
            result[key] = id_list

            tf.logging.info(f"{key}: {len(id_list)}")
            ttl += len(id_list)

        tf.logging.info(f"Total unique files found: {ttl}")

        return result
        
    def mode_size(self, mode='train'):
        return len(self.data_index[mode])      

    def copy_id(self, id, src='valid'):
        """
            Copy files from given id to given source
        """
        tf.logging.info(f"Copying files with id {id}...")
        search_path = os.path.join(self.src_dir, f"{id}-*.{IMG_EXT}")
        for img_path in gfile.Glob(search_path):
            shutil.copy2(img_path, os.path.join(self.src_dir, '..', src))
    
    def copy_mode(self, mode='valid', src='valid'):
        """
            Copy files from given mode to given source
        """
        for id in self.data_index[mode]:
            self.copy_id(id, src=src)

    def _pad_to_size(self, sample):
        """
            Adds padding to the sample to any side that is less than the image size. Original image size
            and adjustments are saved and returned in a dict
        """
        rows, cols, _ = sample.shape

        top_adj = bot_adj = left_adj = right_adj = 0
        if rows < self.img_size:
            diff = self.img_size - rows
            top_adj = math.ceil(diff / 2)
            bot_adj = math.floor(diff / 2)

        if cols < self.img_size:
            diff = self.img_size - cols
            left_adj = math.ceil(diff / 2)
            right_adj = math.floor(diff / 2)

        sample = np.pad(sample, ((top_adj, bot_adj), (left_adj, right_adj), (0, 0)), mode=PAD_MODE)

        pad_info = {'rows': rows,
                    'cols': cols,
                    'tb_adj': (top_adj, bot_adj), 'lr_adj': (left_adj, right_adj)}
        tf.logging.debug(f"pad_info: {pad_info}")
        
        return sample, pad_info

    def _sampling_points(self, sample, seed=None):
        """
            Returns the top and left coordinate to sample image from if any image dimension is greater than
            image size
            
            'seed' is a tuple with values on [0,1) used to generate the top and left points
        """
        top_max = sample.shape[0] - self.img_size + 1
        left_max = sample.shape[1] - self.img_size + 1

        if seed is not None:
            top = int(seed[0] * top_max)
            left = int(seed[1] * left_max)
            tf.logging.debug(f"Using seed: {seed}")
        else:
            top = np.random.randint(0, top_max)
            left = np.random.randint(0, left_max)
            tf.logging.debug(f"Picking randomly")
        tf.logging.debug(f"top, left: {top}, {left}")
        
        return top, left
    
    def _augment(self, sample, sample_seg, sample_con):
        if np.random.randint(0, 2):
            k = np.random.randint(1, 4)
            sample = np.rot90(sample, k)
            sample_seg = np.rot90(sample_seg, k)
            sample_con = np.rot90(sample_con, k)
            tf.logging.debug(f"Rotated 90 degrees {k} times")

        if np.random.randint(0, 2):
            sample = sample[:, ::-1]
            sample_seg = sample_seg[:, ::-1]
            sample_con = sample_con[:, ::-1]
            tf.logging.debug(f"Mirrored on columns")

        sample = self._color_aug.augment_image(sample)

        # TODO: distortion
        # TODO: brightness
        # TODO: noise - see test set for example of noisy image (note: bright and purple ones appear more consistent re: noise
        return sample, sample_seg, sample_con

    def _labels(self, sample_id, pad_info):
        """
            Get the ground truth from the id and pad as necessary
        """
        sample_seg = np.asarray(Image.open(os.path.join(self.src_dir, f"{sample_id}-{IMG_SEGMENT}.{IMG_EXT}")))
        sample_con = np.asarray(Image.open(os.path.join(self.src_dir, f"{sample_id}-{IMG_CONTOUR}.{IMG_EXT}")))

        sample_seg = np.pad(sample_seg, (pad_info['tb_adj'], pad_info['lr_adj']), mode=PAD_MODE)
        sample_con = np.pad(sample_con, (pad_info['tb_adj'], pad_info['lr_adj']), mode=PAD_MODE)

        return sample_seg, sample_con

    def _valid_crop(self, sample_con):
        """
            Return if a valid crop was attained and the top, left coordinates. Valid crops must meet or exceed the
            configured contour ratio. If all tries are attempted without a hit, the last top, left attempt is returned.
        """
        # Will attempt crops based on the size of the image (larger image will have more attempts)
        CROP_ATTEMPTS_PER = 10
        crop_factor = (math.ceil(sample_con.shape[0] / self.img_size) - 1) + (math.ceil(sample_con.shape[1] / self.img_size) - 1)
        crop_attempts_max = CROP_ATTEMPTS_PER * crop_factor
        if crop_attempts_max == 0:
            crop_attempts_max = 1
        tf.logging.debug(f"Maximum crop attempts: {crop_attempts_max}")

        hit = False
        for i in range(crop_attempts_max):
            top, left = self._sampling_points(sample_con)
            cropped_sample_con = sample_con[top:top + self.img_size, left:left + self.img_size]
            con_ratio = np.sum(cropped_sample_con > 0) / np.prod(cropped_sample_con.shape)
            tf.logging.debug(f"Contour ratio: {con_ratio}")

            if con_ratio >= CONTOUR_FREQ_RATIO:
                tf.logging.debug(f"Contour ratio hit: {top}, {left}")
                hit = True
                break
            else:
                tf.logging.debug(f"Contour ratio miss: {top}, {left}")
        
        return hit, top, left

    def batch(self, size, offset=0, mode='train'):
        """
            Return a batch of data from the given mode, offset by the given amount
        """
        assert(mode == 'train' or mode == 'valid')
        
        is_training = (mode == 'train')

        source_ids = self.data_index[mode]
        if size == -1:
            sample_count = len(source_ids)
        else:
            sample_count = max(0, min(size, len(source_ids) - offset))
            
        # Amount of contour misses allowed in this batch (size - required contour hits)
        contour_misses_allowed = size - math.ceil(CONTOUR_FREQ * size)

        # Initializing return values
        inputs = np.zeros((sample_count, self.img_size, self.img_size, IMG_CHANNELS), dtype=np.float32)
        labels_seg = np.zeros((sample_count, self.img_size, self.img_size), dtype=np.float32)
        labels_con = np.zeros((sample_count, self.img_size, self.img_size), dtype=np.float32)

        cur_idx = 0
        contour_misses = 0
        continues = 0
        while True:
            if cur_idx >= sample_count:
                break
            
            if size == -1 or not is_training:
                sample_index = cur_idx + offset
            else:
                sample_index = np.random.randint(len(source_ids))

            sample_id = source_ids[sample_index]
            tf.logging.debug(f"Using sample: {sample_id}")

            sample_src = np.asarray(Image.open(os.path.join(self.src_dir, f"{sample_id}-{IMG_SRC}.{IMG_EXT}")))
            tf.logging.debug(f"pre-pad sample_src.shape: {sample_src.shape}")
            sample_src, pad_info = self._pad_to_size(sample_src)
            tf.logging.debug(f"post-pad sample_src.shape: {sample_src.shape}")
            if _DEBUG_WRITE_:
                Image.fromarray(sample_src).save(f"/tmp/nuclei/{sample_id}-{IMG_SRC}-pad.{IMG_EXT}")

            # Get the ground truth
            sample_seg, sample_con = self._labels(sample_id, pad_info)
            if _DEBUG_WRITE_:
                Image.fromarray(sample_seg).save(f"/tmp/nuclei/{sample_id}-{IMG_SEGMENT}-pad.{IMG_EXT}")
                Image.fromarray(sample_con).save(f"/tmp/nuclei/{sample_id}-{IMG_CONTOUR}-pad.{IMG_EXT}")

            # Picking a random sample of the image (if it is larger than the provided img_size).
            # Validation data will use deterministic seeds if configured.
            if not is_training and self.valid_seeds is not None:
                seed = self.valid_seeds[sample_index]
                top, left = self._sampling_points(sample_src, seed=seed)
            else:
                # Attempt to fulfill the contour ratio requirement up to a point
                if continues > CONTOUR_CONTINUE_MAX:
                    top, left = self._sampling_points(sample_src)
                else:
                    hit, top, left = self._valid_crop(sample_con)
                    if not hit:
                        contour_misses += 1
                        tf.logging.debug(f"Failed contour ratio on sample: {sample_id}, miss count: {contour_misses}")
                        if contour_misses > contour_misses_allowed:
                            continues += 1
                            tf.logging.debug(f"No more misses left. Trying another sample. Continues left: {CONTOUR_CONTINUE_MAX - continues}")
                            continue

            sample_src = sample_src[top:top + self.img_size, left:left + self.img_size, 0:IMG_CHANNELS]
            sample_seg = sample_seg[top:top + self.img_size, left:left + self.img_size]
            sample_con = sample_con[top:top + self.img_size, left:left + self.img_size]

            # Augment the data if training
            if is_training:
                sample_src, sample_seg, sample_con = self._augment(sample_src, sample_seg, sample_con)
 
                if _DEBUG_WRITE_:
                    Image.fromarray(sample_src).save(f"/tmp/nuclei/{sample_id}-{IMG_SRC}-aug.{IMG_EXT}")
                    Image.fromarray(sample_seg).save(f"/tmp/nuclei/{sample_id}-{IMG_SEGMENT}-aug.{IMG_EXT}")
                    Image.fromarray(sample_con).save(f"/tmp/nuclei/{sample_id}-{IMG_CONTOUR}-aug.{IMG_EXT}")

            # Slim's vgg_preprocessing only does the mean subtraction (not the RGB to BGR)
            sample_src = sample_src - np.asarray(VGG_RGB_MEANS, dtype=np.float32)
            # Masks are black and white (0 and 255). Need to convert to labels.
            sample_seg = sample_seg / 255.
            sample_con = sample_con / 255.

            inputs[cur_idx] = sample_src
            labels_seg[cur_idx] = sample_seg
            labels_con[cur_idx] = sample_con
            tf.logging.debug(f"Sample added at index {cur_idx}")
            
            cur_idx += 1

        return inputs, labels_seg, labels_con

    def batch_test(self, offset=0, overlap_const=2, normalize=True):
        """
            Return a single sample from the test set (no labels). The sample is returned in a 2D array of 'img_size' tiles that
            comes from overlapping segments of the original sample with reflective padding. This array is row by columns and can be
            converted into a batch to run for predictions to later be stitched back together to form a prediction for the 
            original image. A dictionary of the information about the sample is returned to aid in reconstruction. The tiles can be 
            normalized with the parameter.
            
            The overlap_const determines the tile overlap ( 1 - 1/overlap_const is overlap percentage). NOTE: this has not been tested
            for anything other than 2
            
            Ex. shape (4, 2, 256, 256, 3) is 4 rows and two columns of 256x256x3 image tiles
        """
        sample_info = {}
        
        sample_id = self.data_index['test'][offset]
        sample_info['id'] = sample_id

        sample_src = np.asarray(Image.open(os.path.join(self.src_dir, f"{sample_id}-{IMG_SRC}.{IMG_EXT}")))
        if _DEBUG_WRITE_:
            img = Image.fromarray(sample_src)
            img.save(f"/tmp/nuclei/original.png")
        tf.logging.debug(f"sample original shape: {sample_src.shape}")
        sample_info['orig_shape'] = sample_src.shape
        sample_src = sample_src[:, :, 0:IMG_CHANNELS]
        orig_rows, orig_cols, _ = sample_src.shape

        pad_base = int(round(self.img_size * (1 - 1.0/overlap_const)))
        pad_residual_row = pad_base - (orig_rows % pad_base) if orig_rows % pad_base > 0 else 0
        pad_residual_col = pad_base - (orig_cols % pad_base) if orig_cols % pad_base > 0 else 0
        
        pad_row = (pad_base + pad_residual_row // 2, pad_base + (pad_residual_row - pad_residual_row // 2))
        pad_col = (pad_base + pad_residual_col // 2, pad_base + (pad_residual_col - pad_residual_col // 2))
        if sample_src.shape[0] < np.sum(pad_row) or sample_src.shape[1] < np.sum(pad_col):
            tf.logging.info(f"WARNING: Padding size ({pad_row}, {pad_col}) is greater than an image dim ({sample_src.shape[0:2]})")
        sample_info['pad_row'] = pad_row
        sample_info['pad_col'] = pad_col
        
        sample_src = np.pad(sample_src, (pad_row, pad_col, (0, 0)), mode='reflect')
        tf.logging.debug(f"sample padded shape: {sample_src.shape}")
        prows, pcols, _ = sample_src.shape
        
        sample_src = np.asarray(sample_src, dtype=np.float32)
        if normalize:
            # Slim's vgg_preprocessing only does the mean subtraction (not the RGB to BGR)
            sample_src = sample_src - VGG_RGB_MEANS

        step = self.img_size // overlap_const
        tf.logging.debug(f"step: {step}")
        sample_info['step'] = step

        tiles = []
        for i in range(0, prows - step, step):
            tiles.append([])
            for j in range(0, pcols - step, step):
                tile = sample_src[i:i + self.img_size, j:j + self.img_size, :]
                tiles[-1].append(tile)

        tiles = np.asarray(tiles)
        tf.logging.debug(f"tiles.shape: {tiles.shape}")

        if _DEBUG_WRITE_:
            img = Image.fromarray(np.asarray(sample_src, dtype=np.uint8))
            img.save(f"/tmp/nuclei/master.png")
            #tiles_debug = tiles.reshape(tiles.shape[0] * tiles.shape[1], *tiles.shape[2:])

            for i in range(tiles.shape[0]):
                for j in range(tiles.shape[1]):
                    img = Image.fromarray(np.asarray(tiles[i, j, :], dtype=np.uint8))
                    img.save(f"/tmp/nuclei/r{i}c{j}.png")

        return tiles, sample_info
    
    def batch_test_abut(self, offset=0, normalize=True):
        """
            Same as 'batch_test' except tiles are not overlapped but are abut
        """
        sample_info = {}
        
        sample_id = self.data_index['test'][offset]
        sample_info['id'] = sample_id

        sample_src = np.asarray(Image.open(os.path.join(self.src_dir, f"{sample_id}-{IMG_SRC}.{IMG_EXT}")))
        if _DEBUG_WRITE_:
            img = Image.fromarray(sample_src)
            img.save(f"/tmp/nuclei/original.png")
        tf.logging.debug(f"sample original shape: {sample_src.shape}")
        sample_info['orig_shape'] = sample_src.shape
        sample_src = sample_src[:, :, 0:IMG_CHANNELS]
        
        padding_row = (0, 0)
        if sample_src.shape[0] % self.img_size > 0:
            padding_ttl = self.img_size - (sample_src.shape[0] % self.img_size)
            padding_row = (padding_ttl // 2, padding_ttl - padding_ttl // 2)
        
        padding_col = (0, 0)
        if sample_src.shape[1] % self.img_size > 0:
            padding_ttl = self.img_size - (sample_src.shape[1] % self.img_size)
            padding_col = (padding_ttl // 2, padding_ttl - padding_ttl // 2)

        tf.logging.debug(f"padding_row, padding_col: {padding_row}, {padding_col}")
        sample_info['padding_row'] = padding_row
        sample_info['padding_col'] = padding_col
        sample_src = np.pad(sample_src, (padding_row, padding_col, (0, 0)), mode='reflect')

        tf.logging.debug(f"sample padded shape: {sample_src.shape}")
        prows, pcols, _ = sample_src.shape
        
        sample_src = np.asarray(sample_src, dtype=np.float32)
        if normalize:
            # Slim's vgg_preprocessing only does the mean subtraction (not the RGB to BGR)
            sample_src = sample_src - VGG_RGB_MEANS

        tiles = []
        for i in range(0, prows, self.img_size):
            tiles.append([])
            for j in range(0, pcols, self.img_size):
                tile = sample_src[i:i + self.img_size, j:j + self.img_size, :]
                tiles[-1].append(tile)

        tiles = np.asarray(tiles)
        tf.logging.debug(f"tiles.shape: {tiles.shape}")

        return tiles, sample_info
