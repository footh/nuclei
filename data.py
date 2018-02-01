from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.platform import gfile

import os
import shutil
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from skimage import measure

IMG_TYPES = ['*.png']

def _remove_files(src):
    """
        Remove files from src directory (train, test, etc) and sub-directories
    """
    if os.path.isfile(src):
        os.unlink(src)
    elif os.path.isdir(src):
        # map lazy evaluates so must wrap in list to force evaluation
        list(map(_remove_files, [os.path.join(src, fi) for fi in os.listdir(src)])) 

def raw_file_list(src='train'):
    """
        Return a list of files from the raw-data directory for the given src
    """
    result = []
    for type in IMG_TYPES:
        search_path = os.path.join('raw-data', src, '*', 'images', type)
        for img_path in gfile.Glob(search_path):
            result.append(img_path)

    return result

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

def kmeans(img_size=(256,256), clusters=3):
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
     
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(x)
    for i in range(clusters):
        lbl_count = np.sum(kmeans.labels_==i)
        print(f"cluster {i}: {lbl_count}")
        print(f"cluster {i} ratio: {lbl_count/len(imgs)}")

    return x, kmeans

def _draw_contours(src, dest):
    """
        Finds contours on src array and writes them to dest array
    """
    contours = measure.find_contours(src, 0.5) # TODO: investigate this parameter
    
    for contour in contours:
        contour = contour.astype(int)
        dest[contour[:,0], contour[:,1]] = 255
        
    return dest
    
def full_mask(raw_file_path, with_contours=True):
    """
        Given the raw_file_path, get the full ground truth mask for that file combining individual masks. If the with_contours
        value is True (default) a contour image will be created.
    """
    masks = []
    for type in IMG_TYPES:    
        search_path = os.path.join(os.path.dirname(raw_file_path), '..', 'masks', type)
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
    
def setup(src='train'):
    """
        Move raw files with full ground truth segment and contour masks over to the src directory. File names are id-{type}.png where 
        type is 'src' for source image, 'seg' for segment mask and 'con' for contour mask
    """
    print(f"Clearing {src} directory...")
    _remove_files(src)

    flist = raw_file_list(src=src)
    for f in flist:
        mask, contour = full_mask(f)
        name, ext = os.path.basename(f).split('.')
        print(f"Processing {name}...")
        
        shutil.copy2(f, os.path.join(src, f"{name}-src.{ext}"))        
        mask.save(os.path.join(src, f"{name}-seg.{ext}"))
        contour.save(os.path.join(src, f"{name}-con.{ext}"))

        