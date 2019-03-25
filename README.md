# Translation-Invariant Attacks

## Introduction
This repository contains the code for [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks]() (CVPR 2019 Oral).

## Method
We proposed a translation-invariant attack method to generate more transferable adversarial examples. This method is implemented by convolving the gradient with a pre-defined kernel in each attack iteration, and can be integrated into any gradient-based attack method. We consider eight STOA defense models on ImageNet

* Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub> trained by [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204)
* High-level representation guided denoiser (HGD, top-1 submission in the NIPS 2017 defense competition)
* Input transformation through random resizing and padding (R&P, rank-2 submission in the NIPS 2017 defense competition)
* Input transformation through JPEG compression or total variance minimization (TVM)
* Rank-3 submission3in the NIPS 2017 defense competition (NIPS-r3)

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
