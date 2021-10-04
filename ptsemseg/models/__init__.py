import copy
import torchvision.models as models

from ptsemseg.models.fcn import fcn8s, fcn16s, fcn32s
from ptsemseg.models.segnet import segnet
from ptsemseg.models.unet import unet
from ptsemseg.models.pspnet import pspnet
from ptsemseg.models.icnet import icnet
from ptsemseg.models.linknet import linknet
from ptsemseg.models.frrn import frrn
from ptsemseg.models.tlcnet import TLCNet, TLCNetU, TLCNetUmux, TLCNetUtlc, TLCNetUtlcmux

def get_model(model_dict, n_maxdisp=256, n_classes=1, version=None):
    name = model_dict["arch"]
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop("arch")

    if name in ["frrnA", "frrnB"]:
        model = model(n_classes, **param_dict)

    elif name in ["fcn32s", "fcn16s", "fcn8s"]:
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "segnet":
        model = model(n_classes=n_classes, **param_dict)
        vgg16 = models.vgg16(pretrained=True)
        model.init_vgg16_params(vgg16)

    elif name == "unet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "pspnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnet":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "icnetBN":
        model = model(n_classes=n_classes, **param_dict)

    elif name == "tlcnet":
        model = model(maxdisp=n_maxdisp, **param_dict)

    elif name == "tlcnetu":
        model = model(n_classes=n_classes, **param_dict)

    elif name=="tlcnetumux": # 2020.10.3 add
        model = model(n_classes=n_classes, **param_dict)

    elif name=="tlcnetutlc": # 2020.10.3 add
        model = model(n_classes=n_classes, **param_dict)

    elif name=="tlcnetutlcmux": # 2020.10.5 add
        model = model(n_classes=n_classes, **param_dict)

    else:
        model = model(n_classes=n_classes, **param_dict)

    return model


def _get_model_instance(name):
    try:
        return {
            "fcn32s": fcn32s,
            "fcn8s": fcn8s,
            "fcn16s": fcn16s,
            "unet": unet,
            "segnet": segnet,
            "pspnet": pspnet,
            "icnet": icnet,
            "icnetBN": icnet,
            "linknet": linknet,
            "frrnA": frrn,
            "frrnB": frrn,
            "tlcnet": TLCNet,
            "tlcnetu": TLCNetU,
            "tlcnetumux": TLCNetUmux,
            "tlcnetutlc": TLCNetUtlc,
            "tlcnetutlcmux": TLCNetUtlcmux
        }[name]
    except:
        raise ("Model {} not available".format(name))
