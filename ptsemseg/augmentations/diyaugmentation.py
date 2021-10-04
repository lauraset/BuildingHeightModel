'''
custom definition
'''
import random
import numpy as np
from skimage.transform import  rotate
from skimage.exposure import rescale_intensity

# from scipy.misc import imrotate
# import cv2

def my_segmentation_transforms(image, segmentation):
    # rotate
    if random.random() > 0.5:
        angle = (np.random.randint(11) + 1) * 15
        # angles=[30, 60, 120, 150, 45, 135] old ones
        # angle = random.choice(angles)
        image = rotate(image, angle) # 1: Bi-linear (default)
        segmentation = rotate(segmentation, angle, order=0) # Nearest-neighbor
        # segmentation = imrotate(segmentation, angle, interp='nearest') # old ones

    #flip left-right
    if random.random() > 0.5:
        image = np.fliplr(image)
        segmentation = np.fliplr(segmentation)

    #flip up-down
    if random.random() > 0.5:
        image = np.flipud(image)
        segmentation = np.flipud(segmentation)

    # brightness
    ratio=random.random()
    if  ratio>0.5:
        image = rescale_intensity(image, out_range=(0, ratio)) #(0.5, 1)

    return image, segmentation


def my_segmentation_transforms_crop(image, segmentation, th):
    # random crop
    h=image.shape[0]
    offset=h-th
    x1 = random.randint(0, offset)
    y1 = random.randint(0, offset)
    image = image[x1:x1+th, y1:y1+th,:]
    segmentation = segmentation[x1:x1+th, y1:y1+th]

    # rotate
    if random.random() > 0.5:
        angles=[30, 60, 120, 150, 45, 135]
        angle = random.choice(angles)
        image = rotate(image, angle)
        segmentation = rotate(segmentation, angle, order=0) # Nearest-neighbor
        # segmentation = imrotate(segmentation, angle, interp='nearest') # old ones

    #flip left-right
    if random.random() > 0.5:
        image = np.fliplr(image)
        segmentation = np.fliplr(segmentation)

    #flip up-down
    if random.random() > 0.5:
        image = np.flipud(image)
        segmentation = np.flipud(segmentation)

    return image, segmentation