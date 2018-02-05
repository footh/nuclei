from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import measure
import hashlib

IMG_TYPES = ['*.png']

MAX_NUM_PER_CLASS = 2**27 - 1  # ~134M

IMG_SRC = 'src'
IMG_CONTOUR = 'con'
IMG_SEGMENT = 'seg'

# TODO: environment variable
tf.logging.set_verbosity(tf.logging.DEBUG)


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
    for type in IMG_TYPES:
        search_path = os.path.join(src_dir, type)
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
    flist = file_list(src=src)
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
    for img_type in IMG_TYPES:
        search_path = os.path.join(os.path.dirname(raw_file_path), '..', 'masks', img_type)
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


def setup(src='train'):
    """
        Move raw files with full ground truth segment and contour masks over to the src directory. File names
        are id-{type}.png where type is 'src' for source image, 'seg' for segment mask and 'con' for contour mask
    """
    print(f"Clearing {src} directory...")
    _remove_files(src)

    flist = raw_file_list(src=src)
    for f in flist:
        mask, contour = full_mask(f)
        name, ext = os.path.basename(f).split('.')
        print(f"Processing {name}...")
        
        shutil.copy2(f, os.path.join(src, f"{name}-{IMG_SRC}.{ext}"))
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
    
    def __init__(self, src='train', img_size=256, validation_pct=15, testing_pct=0):
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

    def batch(self, size, offset=0, mode='train'):
        """
            Return a batch of data from the given mode, offset by the given amount
        """

        source_ids = self.data_index[mode]
        if size == -1:
            sample_count = len(source_ids)
        else:
            sample_count = max(0, min(size, len(source_ids) - offset))

        # Initializing return values
        data = np.zeros((sample_count, self.img_size, self.img_size, 3), dtype=np.float32)
        s_labels = np.zeros((sample_count, self.img_size, self.img_size), dtype=np.float32)
        c_labels = np.zeros((sample_count, self.img_size, self.img_size), dtype=np.float32)

        for i in range(offset, offset + sample_count):
            if size == -1 or mode != 'train':
                sample_index = i
            else:
                sample_index = np.random.randint(len(source_ids))

            sample = source_ids[sample_index]
            tf.logging.debug(f"Using sample: {sample}")

            # TODO: assuming .png here, should just remove the multiple format support
            sample_src = np.asarray(Image.open(os.path.join(self.src, f"{sample}-{IMG_SRC}.png")))
            sample_seg = np.asarray(Image.open(os.path.join(self.src, f"{sample}-{IMG_SEGMENT}.png")))
            sample_con = np.asarray(Image.open(os.path.join(self.src, f"{sample}-{IMG_CONTOUR}.png")))

            # Picking a random sample of the image (if it is larger than the provided img_size)
            top, left = self._sampling_points(sample_src)
            # TODO: should save images without the alpha channel in the setup method
            sample_src = sample_src[top:top + self.img_size, left:left + self.img_size, 0:3]
            sample_seg = sample_seg[top:top + self.img_size, left:left + self.img_size]
            sample_con = sample_con[top:top + self.img_size, left:left + self.img_size]

            # TODO: preprocessing (mean subtraction, RGB to BGR, etc.)
            # TODO: augmentation (flipping, elastic distortion, etc.)

            data[i - offset] = sample_src
            s_labels[i - offset] = sample_seg
            c_labels[i - offset] = sample_con

        return data, s_labels, c_labels
