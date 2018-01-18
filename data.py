from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.platform import gfile

import os
import imageio
import numpy as np

def file_list(src='train'):
    result = []
    IMG_TYPE = ['*.png', '*.jpg'] 
    for type in IMG_TYPE:
        search_path = os.path.join(src, '*', 'images', type)
        for img_path in gfile.Glob(search_path):
            result.append(img_path)

    return result

def as_numpy_arrays(src='train'):
    result = []
    flist = file_list(src=src)
    for f in flist:
        imga = np.asarray(imageio.imread(f))
        #result.append(imga.shape)
        result.append(imga)
    
    return np.asarray(result)