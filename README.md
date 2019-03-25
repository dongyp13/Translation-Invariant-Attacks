# Translation-Invariant Attacks

## Introduction
This repository contains the code for [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks]() (CVPR 2019 Oral).

## Method
We propose a translation-invariant attack method to generate more transferable adversarial examples. Our attack method can fool eight SOTA defense methods on ImageNet at an 82% success rate on average.

Basically, the update rule of momentum iterative method is:

![equation](http://latex.codecogs.com/gif.latex?g_%7Bt&plus;1%7D%20%3D%20%5Cmu%20%5Ccdot%20g_%7Bt%7D%20&plus;%20%5Cfrac%7B%5Cnabla_%7Bx%7DJ%28x_%7Bt%7D%5E%7B*%7D%2Cy%29%7D%7B%5C%7C%5Cnabla_%7Bx%7DJ%28x_%7Bt%7D%5E%7B*%7D%2Cy%29%5C%7C_1%7D%2C%20%5Cquad%20x_%7Bt&plus;1%7D%5E%7B*%7D%20%3D%20%5Cmathrm%7Bclip%7D%28x_%7Bt%7D%5E%7B*%7D%20&plus;%20%5Calpha%5Ccdot%5Cmathrm%7Bsign%7D%28g_%7Bt&plus;1%7D%29%29)


### Citation
If you use momentum iterative method for attacks in your research, please consider citing
the
    @inproceedings{dong2018boosting,
      title={Boosting Adversarial Attacks with Momentum},
      author={Dong, Yinpeng and Liao, Fangzhou and Pang, Tianyu and Su, Hang and Zhu, Jun and Hu, Xiaolin and Li, Jianguo},
      booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
      year={2018}
    }

## Implementation

### Models
We use the ensemble of eight models in our submission, many of which are adversarially trained models. The models can be downloaded [here](http://ml.cs.tsinghua.edu.cn/~yinpeng/nips17/nontargeted/models.zip).

If you want to attack other models, you can replace the model definition part to your own models.

### Cleverhans
We also implement this method in [Cleverhans](https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks.py#L454-L605).

### Targeted Attacks
Please find the targeted attacks at [https://github.com/dongyp13/Targeted-Adversarial-Attack](https://github.com/dongyp13/Targeted-Adversarial-Attack).
