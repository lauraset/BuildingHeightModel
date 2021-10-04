'''
new files yinxcao
used for ningbo high-resolution images
format: png
April 25, 2020
'''

import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import torch
import tifffile as tif
from ptsemseg.augmentations.diyaugmentation import my_segmentation_transforms, my_segmentation_transforms_crop
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif' #new added
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def default_loader(path):
    return np.asarray(Image.open(path))


def stretch_img(image, nrange):
    #according the range [low,high] to rescale image
    h, w, nbands = np.shape(image)
    image_stretch = np.zeros(shape=(h, w, nbands), dtype=np.float32)
    for i in range(nbands):
        image_stretch[:, :, i] = 1.0*(image[:, :, i]-nrange[1, i])/(nrange[0, i]-nrange[1, i])
    return image_stretch


def gray2rgb(image):
    res=np.zeros((image.shape[0], image.shape[1], 3))
    res[ :, :, 0] = image.copy()
    res[ :, :, 1] = image.copy()
    res[ :, :, 2] = image.copy()
    return res


def readtif(name):
    # is gray
    # should be fused with spectral bands
    img=tif.imread(name)
    return img


class myImageFloderold(data.Dataset):
    def __init__(self, imgpath, labpath):  # data loader #params nrange
        self.imgpath = imgpath
        self.labpath = labpath

    def __getitem__(self, index):
        imgpath_ = self.imgpath[index]
        labpath_ = self.labpath[index]
        img = default_loader(imgpath_)
        lab = default_loader(labpath_) #  0, 1, ..., N_CLASS-1
        img = img[:, :, ::-1] / 255  # RGB => BGR
        img = torch.tensor(img, dtype=torch.float).permute(2, 0, 1) # H W C => C H W
        lab = torch.tensor(lab, dtype=torch.long)-1
        return img, lab  # new added

    def __len__(self):
        return len(self.imgpath)

# img, tlc, lab 2020.7.27
# update: 2020.9.11: lab, the unit has been changed to meters (float) rather than floor number.
class myImageFloder(data.Dataset):
    def __init__(self, imgpath, labpath, augmentations=False):  # data loader #params nrange
        self.imgpath = imgpath
        self.labpath = labpath
        self.augmentations = augmentations

    def __getitem__(self, index):
        muxpath_ = self.imgpath[index, 0]
        tlcpath_ = self.imgpath[index, 1]
        labpath_ = self.labpath[index]
        mux = tif.imread(muxpath_) / 10000 # convert to surface reflectance (SR): 0-1
        # tlc = tif.imread(tlcpath_)/950   # stretch to 0-1
        tlc = tif.imread(tlcpath_) / 10000 # convert to 0-1
        img = np.concatenate((mux, tlc), axis=2)  # the third dimension
        img[img>1]=1 # ensure data range is 0-1
        lab = tif.imread(labpath_)  # building floor * 3 (meters) in float format

        if self.augmentations:
            img, lab = my_segmentation_transforms(img, lab)

        img = img.transpose((2, 0, 1)) # H W C => C H W
        # lab = lab.astype(np.int16) * 3 : storing the number of floor, deprecated
        lab = np.expand_dims(lab, axis=0)

        img = torch.tensor(img.copy(), dtype=torch.float)
        lab = torch.tensor(lab.copy(), dtype=torch.float)
        return img, lab

    def __len__(self):
        return len(self.imgpath)


# only load tlc (3bands)
class myImageFloder_tlc(data.Dataset):
    def __init__(self, imgpath, labpath, augmentations=False):  # data loader #params nrange
        self.imgpath = imgpath
        self.labpath = labpath
        self.augmentations = augmentations

    def __getitem__(self, index):
        tlcpath_ = self.imgpath[index, 1]
        labpath_ = self.labpath[index]
        img = tif.imread(tlcpath_)/10000   # stretch to 0-1
        img[img>1] = 1 # ensure data range is 0-1
        lab = tif.imread(labpath_)  # building floor * 3 (meters) in float format

        if self.augmentations:
            img, lab = my_segmentation_transforms(img, lab)

        img = img.transpose((2, 0, 1)) # H W C => C H W
        lab = np.expand_dims(lab, axis=0)

        img = torch.tensor(img.copy(), dtype=torch.float)
        lab = torch.tensor(lab.copy(), dtype=torch.float)
        return img, lab

    def __len__(self):
        return len(self.imgpath)


# only load mux (4 bands) and lab
class myImageFloder_mux(data.Dataset):
    def __init__(self, imgpath, labpath, augmentations=False):  # data loader #params nrange
        self.imgpath = imgpath
        self.labpath = labpath
        self.augmentations = augmentations

    def __getitem__(self, index):
        muxpath_ = self.imgpath[index, 0]
        tlcpath_ = self.imgpath[index, 1]
        labpath_ = self.labpath[index]
        img = tif.imread(muxpath_) / 10000 # convert to surface reflectance (SR): 0-1
        # tlc = tif.imread(tlcpath_)/950   # stretch to 0-1
        # tlc = tif.imread(tlcpath_) / 10000 # convert to 0-1
        # img = np.concatenate((mux, tlc), axis=2)  # the third dimension
        img[img>1]=1 # ensure data range is 0-1
        lab = tif.imread(labpath_)  # building floor * 3 (meters) in float format

        if self.augmentations:
            img, lab = my_segmentation_transforms(img, lab)

        img = img.transpose((2, 0, 1)) # H W C => C H W
        # lab = lab.astype(np.int16) * 3 : storing the number of floor, deprecated
        lab = np.expand_dims(lab, axis=0)

        img = torch.tensor(img.copy(), dtype=torch.float)
        lab = torch.tensor(lab.copy(), dtype=torch.float)
        return img, lab

    def __len__(self):
        return len(self.imgpath)

# img, tlc, lab 2020.8.3
'''
class myImageFloder_tlc(data.Dataset):
    def __init__(self, imgpath, labpath, patchsize=256, augmentations=False):  # data loader #params nrange
        self.imgpath = imgpath
        self.labpath = labpath
        self.patchsize = patchsize
        self.augmentations = augmentations

    def __getitem__(self, index):
        muxpath_ = self.imgpath[index, 0]
        tlcpath_ = self.imgpath[index, 1]
        labpath_ = self.labpath[index]
        mux = tif.imread(muxpath_)  # convert to surface reflectance (SR): 0-1
        tlc = tif.imread(tlcpath_)  # convert to 0-1
        lab = tif.imread(labpath_)  # building floor * 3 (meters)
        # 1. clip
        # random crop: test and train is the same
        offset = mux.shape[0] - self.patchsize
        x1 = random.randint(0, offset)
        y1 = random.randint(0, offset)
        mux = mux[x1:x1 + self.patchsize, y1:y1 + self.patchsize, :]/ 10000
        tlc = tlc[x1:x1 + self.patchsize, y1:y1 + self.patchsize, :] / 10000
        lab = lab[x1:x1 + self.patchsize, y1:y1 + self.patchsize]

        # 2. normalize
        #img = np.concatenate((mux, gray2rgb(tlc[:,:,0]), gray2rgb(tlc[:,:,1]), gray2rgb(tlc[:,:,2])), axis=2)
        img = np.concatenate((mux, tlc), axis=2)
        #img[img>1]=1 # ensure data range is 0-1

        if self.augmentations:
            img, lab = my_segmentation_transforms(img, lab)

        img = img.transpose((2, 0, 1))
        lab = lab.astype(np.int16) * 3

        img = torch.tensor(img.copy(), dtype=torch.float)  #H W C => C H W
        lab = torch.tensor(lab.copy(), dtype=torch.float)
        return img, lab

    def __len__(self):
        return len(self.imgpath)
'''