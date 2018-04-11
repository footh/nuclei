import shutil
import os
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
from skimage import morphology
from skimage import filters
from scipy.spatial import distance
import data
from datetime import datetime
import cv2

_DEBUG_ = False


def plot_compare(img1, img2, title1='Image1', title2='Image2'):
    
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(32, 16), sharex=True, sharey=True)

    ax1.imshow(img1, cmap=plt.cm.gray)
    ax1.set_title(title1)
    ax1.axis('off')
    ax1.set_adjustable('box-forced')
    ax2.imshow(img2, cmap=plt.cm.gray)
    ax2.set_title(title2)
    ax2.axis('off')
    ax2.set_adjustable('box-forced')
    plt.show()


def plot_hist(data, bins=50, ticks=(0, 1000, 20)):
    
    plt.figure(figsize=(15, 15))
    n, bins, patches = plt.hist(data, bins)
    #plt.xlabel('Perimeter Size')
    #plt.ylabel('Frequency')
    #plt.title('Histogram of Perimeter Size')
    major_ticks = np.arange(*ticks)        
    ax = plt.gca()
    ax.set_xticks(major_ticks)
    plt.xticks(rotation=90)    
    plt.grid(True)
    plt.show()


def draw_long_line(img, p1, p2, color=0, size=2):
    """
        Draw line starting at p1 through p2 to the end of the image
    """
    theta = np.arctan2(p1[1]-p2[1], p1[0]-p2[0])
    endpt_x = int(p1[0] - img.shape[0]*np.cos(theta))
    endpt_y = int(p1[1] - img.shape[1]*np.sin(theta))

    cv2.line(img, (p1[0], p1[1]), (endpt_x, endpt_y), color, size)


def area_triangle(p1, p2, p3):
    """
        Calculate the area of a triangle given three vertices
    """
    side1 = distance.euclidean(p1, p2)
    side2 = distance.euclidean(p2, p3)
    side3 = distance.euclidean(p1, p3)

    s = (side1 + side2 + side3) / 2
    area = (s * (s - side1) * (s - side2) * (s - side3)) ** 0.5
    return area


_START_TIME_ = None
def start():
    global _START_TIME_
    _START_TIME_ = datetime.now()


def elapsed():
    result = (datetime.now() - _START_TIME_).total_seconds() * 1000
    start()
    return result

def _debug_output(dp, sample_id, result_seg, result_con, divisors):
    print(f"seg: {np.sum(result_seg > 1)}")
    print(f"con: {np.sum(result_con > 1)}")

    seg_name = f"{sample_id}-{data.IMG_SEGMENT}.type.{data.IMG_EXT}"
    con_name = f"{sample_id}-{data.IMG_CONTOUR}.type.{data.IMG_EXT}"
    
    dp.copy_id(sample_id, src='debug')

    result_seg_out = np.asarray(result_seg * 255, dtype=np.uint8)
    result_con_out = np.asarray(result_con * 255, dtype=np.uint8)
    Image.fromarray(result_seg_out).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'pred')))
    Image.fromarray(result_con_out).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'pred')))
    
    selem = morphology.disk(3)

    # Pipeline #1: Preds -> Close -> Thresh -> Diff
    result_seg_c = morphology.closing(result_seg, selem)
    result_con_c = morphology.closing(result_con, selem)

    result_seg_c = np.asarray(result_seg_c * 255, dtype=np.uint8)
    result_con_c = np.asarray(result_con_c * 255, dtype=np.uint8)
    Image.fromarray(result_seg_c).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'predc1')))
    Image.fromarray(result_con_c).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'predc1')))

    prob_seg_c = filters.threshold_otsu(result_seg_c)
    prob_con_c = filters.threshold_otsu(result_con_c)
    print(f"thresh_seg, thresh_con: {prob_seg_c}, {prob_con_c}")

    thresh_seg_c1 = np.asarray(result_seg_c > prob_seg_c, dtype=np.uint8) * 255
    thresh_con_c1 = np.asarray(result_con_c > prob_con_c, dtype=np.uint8) * 255
    Image.fromarray(thresh_seg_c1).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'thr1')))
    Image.fromarray(thresh_con_c1).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thr1')))

    thresh_con_c1e = morphology.erosion(thresh_con_c1, morphology.square(2))
    Image.fromarray(thresh_con_c1e).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thr1e')))

    diff_1 = np.maximum(thresh_seg_c1 - thresh_con_c1e, 0)
    Image.fromarray(diff_1).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('con.type', 'diff1')))

    # Pipeline #2: Preds -> Thresh -> Close -> Diff
    prob_seg = filters.threshold_otsu(result_seg)
    prob_con = filters.threshold_otsu(result_con)
    print(f"thresh_seg, thresh_con: {prob_seg}, {prob_con}")

    thresh_seg = np.asarray(result_seg > prob_seg, dtype=np.uint8) * 255
    thresh_con = np.asarray(result_con > prob_con, dtype=np.uint8) * 255
    Image.fromarray(thresh_seg).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'thr2')))
    Image.fromarray(thresh_con).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thr2')))

    thresh_seg_c2 = morphology.closing(thresh_seg, selem)
    thresh_con_c2 = morphology.closing(thresh_con, selem)
    Image.fromarray(thresh_seg_c2).save(os.path.join(dp.src_dir, '..', 'debug', seg_name.replace('type', 'thrc2')))
    Image.fromarray(thresh_con_c2).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('type', 'thrc2')))
    diff_2 = np.maximum(thresh_seg_c2 - thresh_con_c2, 0)
    Image.fromarray(diff_2).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('con.type', 'diff2')))

    #m = np.max(divisors)
    #divisors = np.asarray((divisors / m) * 255, dtype=np.uint8)
    #Image.fromarray(divisors).save(os.path.join(dp.src_dir, '..', 'debug', con_name.replace('con.type', 'div')))    