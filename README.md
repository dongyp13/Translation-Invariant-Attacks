# Translation-Invariant Attacks

## Introduction
This repository contains the code for [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks]() (CVPR 2019 Oral).

## Method
We proposed a translation-invariant (TI) attack method to generate more transferable adversarial examples. This method is implemented by convolving the gradient with a pre-defined kernel in each attack iteration, and can be integrated into any gradient-based attack method. 

## Results

We consider eight STOA defense models on ImageNet:

* Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub> trained by [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204);
* [High-level representation guided denoiser](https://arxiv.org/abs/1712.02976) (HGD, top-1 submission in the NIPS 2017 defense competition);
* [Input transformation through random resizing and padding](https://arxiv.org/abs/1711.01991) (R&P, rank-2 submission in the NIPS 2017 defense competition);
* [Input transformation through JPEG compression or total variance minimization (TVM)](https://openreview.net/pdf?id=SyJ7ClWCb);
* [Rank-3 submission3in the NIPS 2017 defense competition (NIPS-r3)](https://github.com/anlthms/nips-2017/tree/master/mmd);

We attacked these models by the fast gradient sign method (FGSM), momentum iterative fast gradient sign method (MI-FGSM), diverse input method (DIM), and their translation-invariant versions as TI-FGSM, TI-MI-FGSM, and TI-DIM. We attacked the ensemble of Inception V3, Inception V4, Inception ResNet V2, and ResNet V2 152. The results are:

<img src="https://github.com/dongyp13/Translation-Invariant-Attacks/rsults.png">

### Citation
If you use our method for attacks in your research, please consider citing
the
    @inproceedings{dong2019evading,
      title={Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks},
      author={Dong, Yinpeng and Pang, Tianyu and Su, Hang and Zhu, Jun},
      booktitle={Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
      year={2019}
    }

## Implementation

### Models
We use the ensemble of eight models in our submission, many of which are adversarially trained models. The models can be downloaded [here](http://ml.cs.tsinghua.edu.cn/~yinpeng/nips17/nontargeted/models.zip).

If you want to attack other models, you can replace the model definition part to your own models.

### Cleverhans
We also implement this method in [Cleverhans](https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks.py#L454-L605).

### Targeted Attacks
Please find the targeted attacks at [https://github.com/dongyp13/Targeted-Adversarial-Attack](https://github.com/dongyp13/Targeted-Adversarial-Attack).
