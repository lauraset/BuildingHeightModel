import os
import numpy as np
from os.path import join

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.tif'  #new added
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath):
    image = list()
    labels = list()

    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        for root, dirs, files in os.walk(filepath):
            if root.endswith('img'):
                for name in files:
                    ipath = os.path.join(root, name)
                    image.append(ipath)
            elif root.endswith('lab'):
                for name in files:
                    ipath = os.path.join(root, name)
                    labels.append(ipath)
    assert len(image) == len(labels)
    image.sort()
    labels.sort()
    image = np.array(image)
    labels = np.array(labels)
    return image, labels


def dataloaderbh(filepath, split=[0.7, 0.1, 0.2]):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    update: 2020.10.3 split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    num_samples=len(img)
    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)
    # generate sequence
    # load the path
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0]) # the same as floor
    num_val = int(num_samples * split[1])

    train = seq[0:num_train]
    val = seq[num_train:(num_train+num_val)]
    # test = seq[num_train:]

    imgt=np.vstack((img[train], tlc[train])).T
    labt=lab[train]

    imgv=np.vstack((img[val], tlc[val])).T
    labv=lab[val]

    return imgt, labt, imgv, labv


def dataloaderbh_trainall(filepath, split=[0.7, 0.1, 0.2]):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    update: 2020.10.3 split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    num_samples=len(img)
    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)
    # generate sequence
    # load the path
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0]) # the same as floor
    num_val = int(num_samples * split[1])

    train = seq[0:num_train]
    val = seq[num_train:(num_train+num_val)]
    # test = seq[num_train:]

    imgt=np.vstack((img[train], tlc[train])).T
    labt=lab[train]

    nameid = [os.path.basename(ipath)[4:10] for ipath in labt]

    return imgt, labt, nameid


def dataloaderbh_valall(filepath, split=[0.7, 0.1, 0.2]):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    update: 2020.10.3 split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    num_samples=len(img)
    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)
    # generate sequence
    # load the path
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0]) # the same as floor
    num_val = int(num_samples * split[1])

    train = seq[0:num_train]
    val = seq[num_train:(num_train+num_val)]
    # test = seq[num_train:]

    imgv=np.vstack((img[val], tlc[val])).T
    labv=lab[val]


    nameid = [os.path.basename(ipath)[4:10] for ipath in labv]

    return imgv, labv, nameid


# return all images and name
def dataloaderbhall(filepath):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    update: 2020.10.3 split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)

    imgt=np.vstack((img, tlc)).T
    labt=lab

    nameid = [os.path.basename(ipath)[4:10] for ipath in labt]
    return imgt, labt, nameid


# Oct. 6th, 2020
# generate test image and labels
def dataloaderbh_test(filepath, split=[0.7, 0.1, 0.2]):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    update: 2020.10.3 split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    num_samples=len(img)
    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)
    # generate sequence
    # load the path
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0])
    num_val = int(num_samples * split[1])

    # train = seq[0:num_train]
    # val = seq[num_train:(num_train+num_val)]
    test = seq[(num_train+num_val):]

    imgt=np.vstack((img[test], tlc[test])).T
    labt=lab[test]

    return imgt, labt


# return imageid
def dataloaderbh_testall(filepath, split=[0.7, 0.1, 0.2]):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    update: 2020.10.3 split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    num_samples=len(img)
    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)
    # generate sequence
    # load the path
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0])
    num_val = int(num_samples * split[1])

    # train = seq[0:num_train]
    # val = seq[num_train:(num_train+num_val)]
    test = seq[(num_train+num_val):]

    imgt=np.vstack((img[test], tlc[test])).T
    labt=lab[test]
    nameid = [os.path.basename(ipath)[4:10] for ipath in labt]

    return imgt, labt, nameid


# Oct. 6th, 2020
# generate val image and labels
def dataloaderbh_val(filepath, split=[0.7, 0.1, 0.2]):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    update: 2020.10.3 split data into train/test/val=7:1:2
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    num_samples=len(img)
    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)
    # generate sequence
    # load the path
    seqpath = join(filepath, 'seq.txt')
    if os.path.exists(seqpath):
        seq = np.loadtxt(seqpath, delimiter=',')
    else:
        seq = np.random.permutation(num_samples)
        np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
    seq = np.array(seq, dtype='int32')

    num_train = int(num_samples * split[0])
    num_val = int(num_samples * split[1])

    # train = seq[0:num_train]
    # val = seq[num_train:(num_train+num_val)]
    val = seq[num_train:(num_train+num_val)]

    imgt=np.vstack((img[val], tlc[val])).T
    labt=lab[val]

    return imgt, labt


# Oct. 17th, 2020
# generate multi-temporal image and labels for t2
def dataloaderbh_test_tempo2(filepath):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    '''
    if not os.path.exists(filepath):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
        tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
        lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    imgsuffix=[os.path.basename(imgname) for imgname in img]
    imgsuffix=[imgname[4:] for imgname in imgsuffix]

    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)

    return np.vstack((img, tlc)).T, lab, imgsuffix


# Oct. 17th, 2020
# generate multi-temporal image and labels for t1: they are used for constructing train/val/test
def dataloaderbh_test_tempo1(path2, path1):
    '''
    :param filepath: the root dir of img, lab and tlc
    :return: img, tlc, lab
    '''
    if not os.path.exists(path1):
        raise ValueError('The path of the dataset does not exist.')
    else:
        img = [join(path1, 'img', name) for name in os.listdir(join(path2, 'img'))]
        tlc = [join(path1, 'tlc', name) for name in os.listdir(join(path2, 'tlc'))]
        lab = [join(path1, 'lab', name) for name in os.listdir(join(path2, 'lab'))]

    assert len(img) == len(tlc)
    assert len(img) == len(lab)
    img.sort()
    tlc.sort()
    lab.sort()

    imgsuffix=[os.path.basename(imgname) for imgname in img]
    imgsuffix=[imgname[4:] for imgname in imgsuffix]

    img=np.array(img)
    tlc=np.array(tlc)
    lab=np.array(lab)

    return np.vstack((img, tlc)).T, lab, imgsuffix

# def dataloaderbh(filepath, split=0.8):
#     '''
#     :param filepath: the root dir of img, lab and tlc
#     :return: img, tlc, lab
#     '''
#     if not os.path.exists(filepath):
#         raise ValueError('The path of the dataset does not exist.')
#     else:
#         img = [join(filepath, 'img', name) for name in os.listdir(join(filepath, 'img'))]
#         tlc = [join(filepath, 'tlc', name) for name in os.listdir(join(filepath, 'tlc'))]
#         lab = [join(filepath, 'lab', name) for name in os.listdir(join(filepath, 'lab'))]
#
#     assert len(img) == len(tlc)
#     assert len(img) == len(lab)
#     img.sort()
#     tlc.sort()
#     lab.sort()
#
#     num_samples=len(img)
#     img=np.array(img)
#     tlc=np.array(tlc)
#     lab=np.array(lab)
#     # generate sequence
#     # load the path
#     seqpath = join(filepath, 'seq.txt')
#     if os.path.exists(seqpath):
#         seq = np.loadtxt(seqpath, delimiter=',')
#     else:
#         seq = np.random.permutation(num_samples)
#         np.savetxt(seqpath, seq, fmt='%d', delimiter=',')
#     # path
#     seq = np.array(seq, dtype='int32')
#     num_train = int(num_samples * split)
#     train = seq[0:num_train]
#     val = seq[num_train:]
#
#     imgt=np.vstack((img[train], tlc[train])).T
#     labt=lab[train]
#
#     imgv=np.vstack((img[val], tlc[val])).T
#     labv=lab[val]
#
#     return imgt, labt, imgv, labv


def dataloader_tv(filepath):
    trainimg, trainlab = dataloader(os.path.join(filepath,'train'))
    valimg, vallab = dataloader(os.path.join(filepath,'val'))
    return trainimg, trainlab, valimg, vallab