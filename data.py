from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.platform import gfile

import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

def file_list(src='train'):
    result = []
    IMG_TYPE = ['*.png', '*.jpg'] 
    for type in IMG_TYPE:
        search_path = os.path.join(src, '*', 'images', type)
        for img_path in gfile.Glob(search_path):
            result.append(img_path)

    return result

def as_images(src='train', size=None):
    result = []
    flist = file_list(src=src)
    for f in flist:
        img = Image.open(f)
        if size is not None:
            img = img.resize(size)
        result.append(img)
    
    return result

def kmeans(img_size=(256,256), clusters=3):
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
    