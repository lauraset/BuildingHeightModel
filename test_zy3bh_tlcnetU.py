'''
test ningbo images
'''
import os
import yaml
import shutil
import torch
import random
import argparse
import numpy as np

from ptsemseg.models import get_model
from ptsemseg.utils import get_logger
from tensorboardX import SummaryWriter
from ptsemseg.loader.diy_dataset import dataloaderbh
import sklearn.metrics
import matplotlib.pyplot as plt
import tifffile as tif


def main(cfg, writer, logger):

    # Setup device
    device = torch.device(cfg["training"]["device"])

    # Setup Dataloader
    data_path = cfg["data"]["path"]
    n_classes = cfg["data"]["n_class"]
    n_maxdisp = cfg["data"]["n_maxdisp"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    learning_rate = cfg["training"]["learning_rate"]
    patchsize = cfg["data"]["img_rows"]

    _, _, valimg, vallab = dataloaderbh(data_path)

    # Setup Model
    model = get_model(cfg["model"], n_maxdisp=n_maxdisp, n_classes=n_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    #resume = cfg["training"]["resume"]
    resume = r'runs\tlcnetu_zy3bh\V1\finetune_200.tar'
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    model.eval()

    for idx, imgpath in enumerate(valimg[0:20]):
        name = os.path.basename(vallab[idx])
        respath = os.path.join(cfg["savepath"],'pred'+name)
        y_true = tif.imread(vallab[idx])
        y_true = y_true.astype(np.int16)*3
        # random crop: test and train is the same
        mux = tif.imread(imgpath[0])/10000 # convert to surface reflectance (SR): 0-1
        tlc = tif.imread(imgpath[1])/10000   # stretch to 0-1

        offset = mux.shape[0] - patchsize
        x1 = random.randint(0, offset)
        y1 = random.randint(0, offset)
        mux = mux[x1:x1 + patchsize, y1:y1 + patchsize, :]
        tlc = tlc[x1:x1 + patchsize, y1:y1 + patchsize, :]
        y_true = y_true[x1:x1 + patchsize, y1:y1 + patchsize]

        img = np.concatenate((mux, tlc), axis=2)
        img[img > 1] = 1  # ensure data range is 0-1
        # remove tlc
        # img[:,:,4:] = 0

        img = img.transpose((2, 0, 1))
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img).float()
        y_res = model(img.to(device))

        y_pred = y_res[0] # height
        y_pred = y_pred.cpu().detach().numpy()
        y_pred = np.squeeze(y_pred)
        rmse = myrmse(y_true, y_pred)

        y_seg = y_res[1] # seg
        y_seg = y_seg.cpu().detach().numpy()
        y_seg = np.argmax(y_seg.squeeze(), axis=0) # C H W=>  H W
        precision, recall, f1score = metricsperclass(y_true, y_seg, value=1) #
        print('rmse: %.3f, segerror: ua %.3f, pa %.3f, f1 %.3f'%(rmse, precision, recall, f1score))

        # tif.imsave((os.path.join(cfg["savepath"],'mux'+name)), mux)
        # tif.imsave( (os.path.join(cfg["savepath"], 'ref' + name)), y_true)
        # tif.imsave( (os.path.join(cfg["savepath"], 'pred' + name)), y_pred)
        tif.imsave((os.path.join(cfg["savepath"], 'seg' + name)), y_seg.astype(np.uint8))

        #
        # color encode: change to the
        # get color info
        # _, color_values = get_colored_info('class_dict.csv')
        # prediction = color_encode(y_pred, color_values)
        # label = color_encode(y_true, color_values)

        # plt.subplot(131)
        # plt.title('Image', fontsize='large', fontweight='bold')
        # plt.imshow(mux[:, :, 0:3]/1000)
        # plt.subplot(132)
        # plt.title('Ref', fontsize='large', fontweight='bold')
        # plt.imshow(y_true)
        # # plt.subplot(143)
        # # plt.title('Pred', fontsize='large', fontweight='bold')
        # # plt.imshow(prediction)
        # plt.subplot(133)
        # plt.title('Pred %.3f'%scores, fontsize='large', fontweight='bold')
        # plt.imshow(y_pred)
        # plt.savefig(os.path.join(cfg["savepath"], 'fig'+name))
        # plt.close()


def gray2rgb(image):
    res=np.zeros((image.shape[0], image.shape[1], 3))
    res[ :, :, 0] = image.copy()
    res[ :, :, 1] = image.copy()
    res[ :, :, 2] = image.copy()
    return res


def metrics(y_true, y_pred, ignorevalue=0):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    maskid = np.where(y_true!=ignorevalue)
    y_true = y_true[maskid]
    y_pred = y_pred[maskid]
    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    kappa = sklearn.metrics.cohen_kappa_score(y_true, y_pred)
    f1_micro = sklearn.metrics.f1_score(y_true, y_pred, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true, y_pred, average="macro")
    f1_weighted = sklearn.metrics.f1_score(y_true, y_pred, average="weighted")
    recall_micro = sklearn.metrics.recall_score(y_true, y_pred, average="micro")
    recall_macro = sklearn.metrics.recall_score(y_true, y_pred, average="macro")
    recall_weighted = sklearn.metrics.recall_score(y_true, y_pred, average="weighted")
    precision_micro = sklearn.metrics.precision_score(y_true, y_pred, average="micro")
    precision_macro = sklearn.metrics.precision_score(y_true, y_pred, average="macro")
    precision_weighted = sklearn.metrics.precision_score(y_true, y_pred, average="weighted")

    return dict(
        accuracy=accuracy,
        kappa=kappa,
        f1_micro=f1_micro,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        recall_micro=recall_micro,
        recall_macro=recall_macro,
        recall_weighted=recall_weighted,
        precision_micro=precision_micro,
        precision_macro=precision_macro,
        precision_weighted=precision_weighted,
    )

def myrmse(y_true, ypred):
    diff=y_true.flatten()-ypred.flatten()
    return np.sqrt(np.mean(diff*diff))


def metricsperclass(y_true, y_pred, value):
    y_pred = y_pred.flatten()
    y_true = np.where(y_true>0, np.ones_like(y_true), np.zeros_like(y_true)).flatten()

    tp=len(np.where((y_true==value) & (y_pred==value))[0])
    tn=len(np.where(y_true==value)[0])
    fn = len(np.where(y_pred == value)[0])
    precision = tp/(1e-10+fn)
    recall = tp/(1e-10+tn)
    f1score = 2*precision*recall/(precision+recall+1e-10)
    return precision, recall, f1score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/tlcnetu_zy3bh.yml",
        help="Configuration file to use",
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp)

    #run_id = random.randint(1, 100000)
    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], "V1")
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    main(cfg, writer, logger)
