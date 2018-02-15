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
from skimage import morphology
import hashlib
import math
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

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
    tf.gfile.MakeDirs('/tmp/nuclei')
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


def _draw_contours(src, dest, dilation_val=2):
    """
        Finds contours on src array and writes them to dest array
    """
    contours = measure.find_contours(src, 0.5)  # TODO: investigate this parameter

    result = np.zeros(src.shape, dtype=np.uint8)
    for contour in contours:
        contour = contour.astype(int)
        result[contour[:, 0], contour[:, 1]] = 255

    # se = morphology.disk(1)
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
    print(f"Clearing {src} directory...")
    _remove_files(src)

    flist = raw_file_list(src=src)
    for cnt, f in enumerate(flist):
        if early_stop is not None and cnt > early_stop:
            break

        if with_masks:
            mask, contour = full_mask(f)
        
        name, ext = os.path.basename(f).split('.')
        print(f"Processing {name}...")
        
        shutil.copy2(f, os.path.join(src, f"{name}-{IMG_SRC}.{ext}"))
        if with_masks:
            mask.save(os.path.join(src, f"{name}-{IMG_SEGMENT}.{ext}"))
            contour.save(os.path.join(src, f"{name}-{IMG_CONTOUR}.{ext}"))


def ratio(src='train'):
    """
        Retrieve the ratios of segment and contour masks
    """
    seg_pixels = 0
    seg_total = 0

    seg_files = file_list(src, fname_wildcard=f"*-{IMG_SEGMENT}")
    tf.logging.info(f"Segment files found: {len(seg_files)}")
    for seg_file in seg_files:
        seg = np.asarray(Image.open(seg_file))
        seg_total += np.prod(seg.shape)
        seg_pixels += np.sum(seg > 0)

    con_pixels = 0
    con_total = 0

    con_files = file_list(src, fname_wildcard=f"*-{IMG_CONTOUR}")
    tf.logging.info(f"Contour files found: {len(con_files)}")
    for con_file in con_files:
        con = np.asarray(Image.open(con_file))
        con_total += np.prod(con.shape)
        con_pixels += np.sum(con > 0)

    assert(seg_total == con_total)

    seg_ratio = seg_pixels / (seg_total * 2)
    con_ratio = con_pixels / (con_total * 2)

    return seg_ratio, con_ratio


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
        self.src = src
        self.img_size = img_size
        self.validation_pct = validation_pct
        self.testing_pct = testing_pct
        self.data_index = {'train': [], 'valid': [], 'test': []}

        self._generate_data_index()

        self.valid_seeds = None
        if valid_same:
             self.valid_seeds = np.random.rand(self.mode_size(mode='valid'), 2)

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

        # TODO: distortion
        return sample, sample_seg, sample_con

    def _labels(self, sample_id, pad_info, top=0, left=0):
        sample_seg = np.asarray(Image.open(os.path.join(self.src, f"{sample_id}-{IMG_SEGMENT}.{IMG_EXT}")))
        sample_con = np.asarray(Image.open(os.path.join(self.src, f"{sample_id}-{IMG_CONTOUR}.{IMG_EXT}")))

        sample_seg = np.pad(sample_seg, (pad_info['tb_adj'], pad_info['lr_adj']), mode=PAD_MODE)
        sample_con = np.pad(sample_con, (pad_info['tb_adj'], pad_info['lr_adj']), mode=PAD_MODE)
        if _DEBUG_:
            Image.fromarray(sample_seg).save(f"/tmp/nuclei/{sample_id}-{IMG_SEGMENT}-pad.{IMG_EXT}")
            Image.fromarray(sample_con).save(f"/tmp/nuclei/{sample_id}-{IMG_CONTOUR}-pad.{IMG_EXT}")

        sample_seg = sample_seg[top:top + self.img_size, left:left + self.img_size]
        sample_con = sample_con[top:top + self.img_size, left:left + self.img_size]

        return sample_seg, sample_con

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
            tf.logging.debug(f"pre-pad sample_src.shape: {sample_src.shape}")
            sample_src, pad_info = self._pad_to_size(sample_src)
            tf.logging.debug(f"post-pad sample_src.shape: {sample_src.shape}")
            inputs_info.append(pad_info)
            if _DEBUG_:
                Image.fromarray(sample_src).save(f"/tmp/nuclei/{sample_id}-{IMG_SRC}-pad.{IMG_EXT}")
            
            # Picking a random sample of the image (if it is larger than the provided img_size).
            # Validation data will use deterministic seeds if configured.
            seed = None
            if not is_training and self.valid_seeds is not None:
                seed = self.valid_seeds[sample_index]
            top, left = self._sampling_points(sample_src, seed=seed)
            sample_src = sample_src[top:top + self.img_size, left:left + self.img_size, 0:IMG_CHANNELS]

            # Get the ground truth
            sample_seg, sample_con = self._labels(sample_id, pad_info, top, left)

            # Augment the data if training
            if is_training:
                sample_src, sample_seg, sample_con = self._augment(sample_src, sample_seg, sample_con)
 
                if _DEBUG_:
                    Image.fromarray(sample_src).save(f"/tmp/nuclei/{sample_id}-{IMG_SRC}-aug.{IMG_EXT}")
                    Image.fromarray(sample_seg).save(f"/tmp/nuclei/{sample_id}-{IMG_SEGMENT}-aug.{IMG_EXT}")
                    Image.fromarray(sample_con).save(f"/tmp/nuclei/{sample_id}-{IMG_CONTOUR}-aug.{IMG_EXT}")

            # Slim's vgg_preprocessing only does the mean subtraction (not the RGB to BGR)
            sample_src = sample_src - np.asarray(VGG_RGB_MEANS, dtype=np.float32)
            # Masks are black and white (0 and 255). Need to convert to labels.
            sample_seg = sample_seg / 255.
            sample_con = sample_con / 255.

            inputs[i - offset] = sample_src
            labels_seg[i - offset] = sample_seg
            labels_con[i - offset] = sample_con

        return inputs, labels_seg, labels_con

    def batch_test(self, offset=0, overlap_const=2, normalize=True):
        """
            Return a single sample from the test set (no labels). The sample is returned in a 2D array of 'img_size' tiles that
            comes from overlapping segments of the original sample with reflective padding. This array is row by columns and can be
            converted into a batch to run for predictions to later be stitched back together to form a prediction for the 
            original image. A dictionary of the information about the sample is returned to aid in reconstruction. The tiles can be 
            normalized with the parameter.
            
            The overlap_const determines the tile overlap ( 1 - 1/overlap_const is overlap percentage)
            
            Ex. shape (4, 2, 256, 256, 3) is 4 rows and two columns of 256x256x3 image tiles
        """
        sample_info = {}
        
        sample_id = self.data_index['test'][offset]
        sample_info['id'] = sample_id

        sample_src = np.asarray(Image.open(os.path.join(self.src, f"{sample_id}-{IMG_SRC}.{IMG_EXT}")))
        if _DEBUG_:
            img = Image.fromarray(sample_src)
            img.save(f"/tmp/nuclei/original.png")
        tf.logging.debug(f"sample original shape: {sample_src.shape}")
        sample_info['orig_shape'] = sample_src.shape
        sample_src = sample_src[:, :, 0:IMG_CHANNELS]

        padding = int(round(self.img_size * (1 - 1.0/overlap_const)))
        tf.logging.debug(f"padding: {padding}")
        if sample_src.shape[0] < padding or sample_src.shape[1] < padding:
            raise ValueError(f"Padding size ({padding}) cannot be greater than any image dim ({sample_src.shape[0:2]})")
        sample_info['padding'] = padding
        
        sample_src = np.pad(sample_src, ((padding, padding), (padding, padding), (0, 0)), mode='reflect')
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

        return tiles, sample_info
