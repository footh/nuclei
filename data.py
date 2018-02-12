from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import sys
import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import measure
import hashlib
import math

IMG_EXT = 'png'
IMG_CHANNELS = 3
VGG_RGB_MEANS = [123.68, 116.78, 103.94]
PAD_MODE = 'reflect'

MAX_NUM_PER_CLASS = 2**27 - 1  # ~134M

IMG_SRC = 'src'
IMG_CONTOUR = 'con'
IMG_SEGMENT = 'seg'

_DEBUG_ = False


def SET_DEBUG(val=False):
    sys.modules[__name__]._DEBUG_ = val
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


def file_list(src_dir):
    """
        Return a list of files from the src_dir directory
    """
    result = []
    search_path = os.path.join(src_dir, f"*.{IMG_EXT}")
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
    
    for contour in contours:
        contour = contour.astype(int)
        dest[contour[:,0], contour[:, 1]] = 255
        
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
            # TODO: DCAN paper talks about applying a transformation to the contours to make them
            # better (thicker I think?)
            contour = _draw_contours(mask_img, contour)            
        
    return Image.fromarray(mask), Image.fromarray(contour)


def setup(src='train', with_masks=True):
    """
        Move raw files with full ground truth segment and contour masks over to the src directory. File names
        are id-{type}.png where type is 'src' for source image, 'seg' for segment mask and 'con' for contour mask
    """
    print(f"Clearing {src} directory...")
    _remove_files(src)

    flist = raw_file_list(src=src)
    for f in flist:
        if with_masks:
            mask, contour = full_mask(f)
        
        name, ext = os.path.basename(f).split('.')
        print(f"Processing {name}...")
        
        shutil.copy2(f, os.path.join(src, f"{name}-{IMG_SRC}.{ext}"))
        if with_masks:
            mask.save(os.path.join(src, f"{name}-{IMG_SEGMENT}.{ext}"))
            contour.save(os.path.join(src, f"{name}-{IMG_CONTOUR}.{ext}"))
        
        
def which_set(file_id, validation_pct, testing_pct):
    """
        Determines which data partition the file should belong to.
        (taken from tensorflow speech audio tutorial
    
        Returns string, one of 'training', 'validation', or 'testing'.
    """
    file_id_hashed = hashlib.sha1(compat.as_bytes(file_id)).hexdigest()
    percentage_hash = ((int(file_id_hashed, 16) % (MAX_NUM_PER_CLASS + 1)) * (100.0 / MAX_NUM_PER_CLASS))

    if percentage_hash < validation_pct:
        result = 'valid'
    elif percentage_hash < (testing_pct + validation_pct):
        result = 'test'
    else:
        result = 'train'
    
    return result


class DataProcessor:
    
    def __init__(self, src='train', img_size=256, validation_pct=0, testing_pct=0):
        self.src = src
        self.img_size = img_size
        self.validation_pct = validation_pct
        self.testing_pct = testing_pct
        self.data_index = {'train': [], 'valid': [], 'test': []}

        self._generate_data_index()

    def _generate_data_index(self):
        """
            Build the index of files, bucketed by source group
        """
        src_dir = self.src

        # Getting the unique file ids in the src set
        all_files = file_list(src_dir)
        file_ids = list({os.path.basename(file).split('-')[0] for file in all_files})
        tf.logging.info(f"Total unique files found: {len(file_ids)}")
        
        for file_id in file_ids:
            idx = which_set(file_id, self.validation_pct, self.testing_pct)
            self.data_index[idx].append(file_id)
            
        tf.logging.info(f"Training total: {len(self.data_index['train'])}")
        tf.logging.info(f"Validation total: {len(self.data_index['valid'])}")
        tf.logging.info(f"Testing total: {len(self.data_index['test'])}")
        
    def mode_size(self, mode='train'):
        return len(self.data_index[mode])        

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
        tf.logging.debug(f"pre pad sample shape: {sample.shape}")

        sample = np.pad(sample, ((top_adj, bot_adj), (left_adj, right_adj), (0, 0)), mode=PAD_MODE)

        sample_info = {'rows': rows,
                       'cols': cols,
                       'tb_adj': (top_adj, bot_adj), 'lr_adj': (left_adj, right_adj)}
        tf.logging.debug(f"sample_info: {sample_info}")
        
        return sample, sample_info

    def _sampling_points(self, sample):
        """
            Returns the top and left coordinate to sample image from if any image dimension is greater than
            image size
        """
        tf.logging.debug(f"sample shape: {sample.shape}")

        top_max = sample.shape[0] - self.img_size + 1
        left_max = sample.shape[1] - self.img_size + 1

        top = np.random.randint(0, top_max)
        left = np.random.randint(0, left_max)
        tf.logging.debug(f"top, left: {top}, {left}")
        return top, left
    
    def _augment(self, sample):
        # TODO: sample augmentation (random flips, distortion, etc.)
        return sample

    def _labels(self, sample_id, sample_info, top=0, left=0):
        sample_seg = np.asarray(Image.open(os.path.join(self.src, f"{sample_id}-{IMG_SEGMENT}.{IMG_EXT}")))
        sample_con = np.asarray(Image.open(os.path.join(self.src, f"{sample_id}-{IMG_CONTOUR}.{IMG_EXT}")))

        sample_seg = np.pad(sample_seg, (sample_info['tb_adj'], sample_info['lr_adj'], (0, 0)), mode=PAD_MODE)
        sample_con = np.pad(sample_con, (sample_info['tb_adj'], sample_info['lr_adj'], (0, 0)), mode=PAD_MODE)
        if _DEBUG_:
            Image.fromarray(sample_seg).save(f"/tmp/{sample_id}-{IMG_SEGMENT}.{IMG_EXT}")
            Image.fromarray(sample_con).save(f"/tmp/{sample_id}-{IMG_CONTOUR}.{IMG_EXT}")


        sample_seg = sample_seg[top:top + self.img_size, left:left + self.img_size]
        sample_con = sample_con[top:top + self.img_size, left:left + self.img_size]

        # Masks are black and white (0 and 255). Need to convert to labels.
        sample_seg = sample_seg / 255.
        sample_con = sample_con / 255.

        return sample_seg, sample_con

    def batch(self, size, offset=0, mode='train', with_labels=True):
        """
            Return a batch of data from the given mode, offset by the given amount
        """
        is_training = (mode == 'train')

        source_ids = self.data_index[mode]
        if size == -1:
            sample_count = len(source_ids)
        else:
            sample_count = max(0, min(size, len(source_ids) - offset))

        # Initializing return values
        inputs = np.zeros((sample_count, self.img_size, self.img_size, IMG_CHANNELS), dtype=np.float32)
        inputs_info = []
        labels_seg = np.zeros((sample_count, self.img_size, self.img_size), dtype=np.float32)
        labels_con = np.zeros((sample_count, self.img_size, self.img_size), dtype=np.float32)

        for i in range(offset, offset + sample_count):
            if size == -1 or not is_training:
                sample_index = i
            else:
                sample_index = np.random.randint(len(source_ids))

            sample_id = source_ids[sample_index]
            tf.logging.debug(f"Using sample: {sample_id}")

            sample_src = np.asarray(Image.open(os.path.join(self.src, f"{sample_id}-{IMG_SRC}.{IMG_EXT}")))
            sample_src, sample_info = self._pad_to_size(sample_src)
            inputs_info.append(sample_info)
            if _DEBUG_:
                Image.fromarray(sample_src).save(f"/tmp/{sample_id}-{IMG_SRC}.{IMG_EXT}")
            
            # Picking a random sample of the image (if it is larger than the provided img_size)
            top, left = self._sampling_points(sample_src)
            sample_src = sample_src[top:top + self.img_size, left:left + self.img_size, 0:IMG_CHANNELS]

            if is_training:
                sample_src = self._augment(sample_src)

            # Slim's vgg_preprocessing only does the mean subtraction (not the RGB to BGR)
            sample_src = sample_src - np.asarray(VGG_RGB_MEANS, dtype=np.float32)

            inputs[i - offset] = sample_src
            if with_labels:
                sample_seg, sample_con = self._labels(sample_id, sample_info, top, left)
                labels_seg[i - offset] = sample_seg
                labels_con[i - offset] = sample_con

        if with_labels:
            return inputs, inputs_info, labels_seg, labels_con
        else:
            return inputs, inputs_info

    def batch_test(self, offset=0, overlap_const=2, normalize=True):
        """
            Return a single sample from the test set (no labels). The sample is returned a 2D array of 'img_size' tiles that
            comes from overlapping segments of the original sample with reflective padding. This array row by columss and can be
            converted into a batch to run for predictions to later be stitched back together to form a prediction for the 
            original image. The tiles can be normalized with the parameter.
            
            The overlap_const determines the tile overlap ( 1/overlap_const is overlap percentage)
            
            Ex. shape (4, 2, 256, 256, 3) is 4 rows and two columns of 256x256x3 image tiles
        """
        sample_id = self.data_index['test'][offset]

        sample_src = np.asarray(Image.open(os.path.join(self.src, f"{sample_id}-{IMG_SRC}.{IMG_EXT}")))
        if _DEBUG_:
            img = Image.fromarray(sample_src)
            img.save(f"/tmp/nuclei/original.png")
        tf.logging.debug(f"sample original shape: {sample_src.shape}")
        sample_src = sample_src[:, :, 0:IMG_CHANNELS]

        padding = int(round(self.img_size * (1 - 1.0/overlap_const)))
        tf.logging.debug(f"padding: {padding}")
        
        sample_src = np.pad(sample_src, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
        tf.logging.debug(f"sample padded shape: {sample_src.shape}")
        prows, pcols, _ = sample_src.shape
        
        sample_src = np.asarray(sample_src, dtype=np.float32)
        if normalize:
            # Slim's vgg_preprocessing only does the mean subtraction (not the RGB to BGR)
            sample_src = sample_src - VGG_RGB_MEANS

        step = self.img_size // overlap_const
        tf.logging.debug(f"step: {step}")

        tiles = []
        for i in range(0, prows - self.img_size + 1, step):
            tiles.append([])
            for j in range(0, pcols - self.img_size + 1, step):
                tile = sample_src[i:i + self.img_size, j:j + self.img_size, :]
                tiles[-1].append(tile)

        tiles = np.asarray(tiles)
        tf.logging.debug(f"tiles.shape: {tiles.shape}")

        if _DEBUG_:
            img = Image.fromarray(np.asarray(sample_src, dtype=np.uint8))
            img.save(f"/tmp/nuclei/master.png")
            #tiles_debug = tiles.reshape(tiles.shape[0] * tiles.shape[1], *tiles.shape[2:])

            for i in range(tiles.shape[0]):
                for j in range(tiles.shape[1]):
                    img = Image.fromarray(np.asarray(tiles[i, j, :], dtype=np.uint8))
                    img.save(f"/tmp/nuclei/r{i}c{j}.png")

        return tiles
