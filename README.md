<h2 align="center">A deep learning method for building height estimation using high-resolution multi-view imagery over urban areas: A case study of 42 Chinese cities.</h2>



<h5 align="right">by Yinxia Cao, Xin Huang </h5>

---------------------


## Getting Started

#### Requirements:
- pytorch >= 1.8.0 (lower version can also work)
- python >=3.6

### Prepare the training set

see the sample directory. Note that the whole dataset is not available publicly now.

### Train the height model
#### 1. Prepare your dataset
#### 2. edit datapath
```
python train_zy3bh_tlcnetU_loss.py
```

#### 3. Evaluate on test set
see the pretrained model in directory runs/
```
python evaluate.py
```

## TO DO




If there is any issue, please feel free to contact me (email: yinxcao@163.com or yinxcao@whu.edu.cn).
## Citation

If you find this repo useful for your research, please consider citing the paper
```

```
### acknowledgement
@article{mshahsemseg,
    Author = {Meet P Shah},
    Title = {Semantic Segmentation Architectures Implemented in PyTorch.},
    Journal = {https://github.com/meetshah1995/pytorch-semseg},
    Year = {2017}
}

