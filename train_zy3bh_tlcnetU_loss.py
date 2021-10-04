'''
date: 2020.7.27
author: yinxia cao
function: train building height using unet method
@Update: 2020.10.8 uncertainty weighting multi-loss
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import yaml
import shutil
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm

from torch.utils import data
from ptsemseg.models import get_model
from ptsemseg.utils import get_logger
from tensorboardX import SummaryWriter #change tensorboardX
from ptsemseg.loader.diy_dataset import dataloaderbh
from ptsemseg.loader.diyloader import myImageFloder
import torch.nn.functional as F
# from segmentation_models_pytorch_revised import DeepLabV3Plus

def main(cfg, writer, logger):

    # Setup seeds
    torch.manual_seed(cfg.get("seed", 1337))
    torch.cuda.manual_seed(cfg.get("seed", 1337))
    np.random.seed(cfg.get("seed", 1337))
    random.seed(cfg.get("seed", 1337))

    # Setup device
    device = torch.device(cfg["training"]["device"])

    # Setup Dataloader
    data_path = cfg["data"]["path"]
    n_classes = cfg["data"]["n_class"]
    n_maxdisp = cfg["data"]["n_maxdisp"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]

    # Load dataset
    trainimg, trainlab, valimg, vallab = dataloaderbh(data_path)
    traindataloader = torch.utils.data.DataLoader(
        myImageFloder(trainimg, trainlab, True),
        batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    testdataloader = torch.utils.data.DataLoader(
        myImageFloder(valimg, vallab),
        batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    # Setup Model
    # model = DeepLabV3Plus("resnet18", encoder_weights='imagenet' )
    model = get_model(cfg["model"], n_maxdisp=n_maxdisp, n_classes=n_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # print the model
    start_epoch = 0
    resume = cfg["training"]["resume"]
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
                 .format(resume, checkpoint['epoch']))
        start_epoch = checkpoint['epoch']
    else:
        print("=> no checkpoint found at resume")
        print("=> Will start from scratch.")

    # define task-dependent log_variance
    log_var_a = torch.zeros((1,), requires_grad=True)
    log_var_b = torch.zeros((1,), requires_grad=True)
    # log_var_c = torch.tensor(1.) # fix the weight of semantic segmentation
    log_var_c = torch.zeros((1,), requires_grad=True)

    # get all parameters (model parameters + task dependent log variances)
    params = ([p for p in model.parameters()] + [log_var_a] + [log_var_b] + [log_var_c])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    optimizer = torch.optim.Adam(params, lr=cfg["training"]["learning_rate"], betas=(0.9, 0.999))

    criterion = 'rmse' #useless

    for epoch in range(epochs-start_epoch):
        epoch = start_epoch + epoch
        adjust_learning_rate(optimizer, epoch)
        model.train()
        train_loss = list()
        train_mse = 0.
        count = 0
        print_count = 0
        vara = list()
        varb = list()
        varc = list()
        # with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for x, y_true in tqdm(traindataloader):
            x = x.to(device, non_blocking=True)
            y_true = y_true.to(device, non_blocking=True)

            ypred1, ypred2, ypred3, ypred4 = model.forward(x)
            y_truebi = torch.where(y_true > 0, torch.ones_like(y_true), torch.zeros_like(y_true))
            y_truebi = y_truebi.long().view(-1).to(device, non_blocking=True)
            ypred3 = ypred3.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
            loss_mse = F.mse_loss(ypred4 , y_true, reduction='mean').cpu().detach().numpy()
            loss = loss_weight([ypred1, ypred2, ypred3, ypred4],
                               [y_true, y_truebi],
                               [log_var_a.to(device), log_var_b.to(device), log_var_c.to(device)])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().detach().numpy())
            train_mse += loss_mse*x.shape[0]
            count += x.shape[0]

            vara.append(log_var_a.cpu().detach().numpy())
            varb.append(log_var_b.cpu().detach().numpy())
            varc.append(log_var_c.cpu().detach().numpy())

            if print_count%20 ==0:
                print('training loss %.3f, rmse %.3f, vara %.2f, b %.2f, c %.2f' %
                  (loss.item(), np.sqrt(loss_mse), log_var_a, log_var_b, log_var_c))
            print_count += 1

        train_rmse = np.sqrt(train_mse/count)
        # test
        val_rmse = test_epoch(model, criterion,
                              testdataloader, device, epoch)
        print("epoch %d rmse: train %.3f, test %.3f" % (epoch, train_rmse, val_rmse))

        # save models
        if epoch % 2 == 0: # every five internval
            savefilename = os.path.join(logdir, 'finetune_'+str(epoch)+'.tar')
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': np.mean(train_loss),
                'test_loss': np.mean(val_rmse), #*100
            }, savefilename)
        #
        writer.add_scalar('train loss',
                          (np.mean(train_loss)), #average
                          epoch)
        writer.add_scalar('train rmse',
                          (np.mean(train_rmse)), #average
                          epoch)
        writer.add_scalar('val rmse',
                          (np.mean(val_rmse)), #average
                          epoch)
        writer.add_scalar('weight a',
                          (np.mean(vara)), #average
                          epoch)
        writer.add_scalar('weight b',
                          (np.mean(varb)), #average
                          epoch)
        writer.add_scalar('weight c',
                          (np.mean(varc)), #average
                          epoch)
        writer.close()


def adjust_learning_rate(optimizer, epoch):
    if epoch <= 200:
        lr = cfg["training"]["learning_rate"]
    elif epoch <=250:
        lr = cfg["training"]["learning_rate"] * 0.1
    elif epoch <=300:
        lr = cfg["training"]["learning_rate"] * 0.01
    else:
        lr = cfg["training"]["learning_rate"] * 0.025 # 0.0025 before
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr #added


# def rmse(disp, gt):
#     errmap = torch.sqrt(torch.pow((disp - gt), 2).mean())
#     return errmap  # rmse


# def mse(disp, gt):
#     return (disp-gt)**2.

# custom loss
def loss_weight_ori(y_pred, y_true, log_vars):
  loss = 0
  for i in range(len(y_pred)):
    precision = torch.exp(-log_vars[i])
    diff = (y_pred[i]-y_true[i])**2.
    loss += torch.sum(precision * diff + log_vars[i], -1)
  return torch.mean(loss)


def loss_weight(y_pred, y_true, log_vars):
  #loss 0 tlc height
  precision0 = torch.exp(-log_vars[0])
  diff0 = F.mse_loss(y_pred[0],y_true[0],reduction='mean')
  loss0 = diff0*precision0 + log_vars[0]
  #loss 1 mux height
  precision1 = torch.exp(-log_vars[1])
  diff1 = F.mse_loss(y_pred[1], y_true[0], reduction='mean')
  loss1 = diff1*precision1 + log_vars[1]
  #loss 2 mux segmentation
  loss2 = F.cross_entropy(y_pred[2], y_true[1], reduction='mean')
  #loss 3 final height
  precision3 = torch.exp(-log_vars[2])
  diff3 = F.mse_loss(y_pred[3], y_true[0], reduction='mean')
  loss3 = diff3*precision3 + log_vars[2]
  return loss0+loss1+loss3+loss2


def crossentrop(ypred, y_true, device='cuda'):
    y_truebi = torch.where(y_true > 0, torch.ones_like(y_true), torch.zeros_like(y_true))
    y_truebi = y_truebi.long().view(-1).to(device)
    ypred = ypred.transpose(1, 2).transpose(2, 3).contiguous().view(-1, 2)
    return F.cross_entropy(ypred, y_truebi)


def test_epoch(model, criterion, dataloader, device, epoch):
    model.eval()
    with torch.no_grad():
        losses = 0.
        count = 0
        for x, y_true in tqdm(dataloader):
            x = x.to(device, non_blocking =True)
            y_true = y_true.to(device, non_blocking =True)

            y_pred, _ = model.forward(x)
            lossv = F.mse_loss(y_pred, y_true, reduction='mean').cpu().detach().numpy()
            losses += lossv*x.shape[0]
            count += x.shape[0]

        lossfinal = np.sqrt(losses/count)
        print('test error %.3f rmse' % lossfinal)
        return lossfinal


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
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    logdir = os.path.join("runs", os.path.basename(args.config)[:-4], "V1")
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Let the games begin")

    main(cfg, writer, logger)
