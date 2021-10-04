# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class heightacc(object):
    '''
    compute acc
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        #self.r2 = 0 不好计算
        self.mse = 0
        self.se = 0
        self.mae = 0
        #self.mape = 0
        self.count = 0
        self.yrefmean = 0
        self.ypref2 = 0

    def update(self, ypred, yref, num):
        self.se += np.mean(ypred-yref)*num

        self.mae += np.mean(np.abs(ypred-yref))*num

        self.mse += np.mean((ypred-yref)**2)*num

        #self.mape += np.mean(np.abs((ypred-yref)/(1e-8+yref)))*num

        self.yrefmean += np.mean(yref)*num

        self.ypref2 += np.mean(yref**2)*num

        self.count += num

    def getacc(self):
        se = self.se/self.count
        mae = self.mae/self.count
        mse = self.mse/self.count
        #mape = self.mape/self.count
        rmse = np.sqrt(mse)

        yrefmean = self.yrefmean/self.count
        yref2 = self.ypref2/self.count
        r2 = 1 - mse/(yref2 -yrefmean**2)
        return r2, rmse, mae, se
