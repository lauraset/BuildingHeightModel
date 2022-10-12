'''
2020.12.28 validate us samples
'''

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import os
import torch
from tqdm import tqdm
import numpy as np
import tifffile as tif

from torch.utils import data
from ptsemseg.models import TLCNetUtlc
from ptsemseg.loader.diy_dataset import dataloaderbh_testall
from ptsemseg.loader.diyloader import myImageFloder_tlc
from ptsemseg.metrics import heightacc

def main():

    # Setup device
    device = 'cuda'

    # Setup Dataloader
    data_path = r'sample'
    batch_size = 16
    # Load dataset
    testimg, testlab, nameid = dataloaderbh_testall(data_path)

    testdataloader = torch.utils.data.DataLoader(
        myImageFloder_tlc(testimg, testlab, num=16),
        batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Setup Model
    model = TLCNetUtlc(n_classes=1).to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

    # print the model
    start_epoch = 0
    resume = r'runs\tlcnetu_zy3bh_tlc\V1\finetune_298.tar'
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
        return

    model.eval()
    acc = heightacc()
    counts = 0
    respath = os.path.dirname(os.path.dirname(resume)).replace('runs', 'pred')
    if not os.path.exists(respath):
        os.makedirs(respath)

    with torch.no_grad():
        for x, y_true in tqdm(testdataloader):
            y_pred, y_seg = model.forward(x.to(device))
            y_pred = y_pred.cpu().detach().numpy()

            acc.update(y_pred, y_true.numpy(), x.shape[0])

            # save to tif
            y_pred = np.squeeze(y_pred, axis=1) # B H W
            y_seg = np.argmax(y_seg.cpu().numpy(), axis=1).astype(np.uint8) # B H W
            count = x.shape[0]
            names = nameid[counts:counts+count]
            for k in range(count):
                tif.imsave((os.path.join(respath,'pred_'+names[k]+'.tif')), y_pred[k])
                tif.imsave((os.path.join(respath,'seg_'+names[k]+'.tif')), y_seg[k])
                tif.imsave((os.path.join(respath, 'seg_' + names[k] + '_clr.tif')), y_seg[k] * 255)
            counts += count

    res = acc.getacc()
    print('r2, rmse, mae, se')
    print('%.6f %.6f %.6f %.6f' % (res[0], res[1], res[2], res[3]))
    print(res)


if __name__ == "__main__":
    main()
