# Translation-Invariant Attacks

## Introduction
This repository contains the code for [Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks](https://arxiv.org/pdf/1904.02884.pdf) (CVPR 2019 Oral).

## Method
We proposed a translation-invariant (TI) attack method to generate more transferable adversarial examples. This method is implemented by convolving the gradient with a pre-defined kernel in each attack iteration, and can be integrated into any gradient-based attack method. 

## Run the code
First download the [models](#Models). You can also use other models by changing the model definition part in the code.
Then run the following command
```
bash run_attack.py input_dir output_dir
```
where original images are stored in ``input_dir`` with ``.png`` format, and the generated adversarial images are saved in ``output_dir``.
We used the Python 2.7 and Tensorflow 1.12 versions. 

## Results

We consider eight STOA defense models on ImageNet:

* Inc-v3<sub>ens3</sub>, Inc-v3<sub>ens4</sub>, IncRes-v2<sub>ens</sub> trained by [Ensemble Adversarial Training](https://arxiv.org/abs/1705.07204);
* [High-level representation guided denoiser](https://arxiv.org/abs/1712.02976) (HGD, top-1 submission in the NIPS 2017 defense competition);
* [Input transformation through random resizing and padding](https://arxiv.org/abs/1711.01991) (R&P, rank-2 submission in the NIPS 2017 defense competition);
* [Input transformation through JPEG compression or total variance minimization (TVM)](https://openreview.net/pdf?id=SyJ7ClWCb);
* [Rank-3 submission in the NIPS 2017 defense competition (NIPS-r3)](https://github.com/anlthms/nips-2017/tree/master/mmd);

We attacked these models by the [fast gradient sign method](https://arxiv.org/abs/1412.6572) (FGSM), [momentum iterative fast gradient sign method](https://arxiv.org/abs/1710.06081) (MI-FGSM), [diverse input method](https://arxiv.org/abs/1803.06978) (DIM), and their translation-invariant versions as TI-FGSM, TI-MI-FGSM, and TI-DIM. We generated adversarial examples for the ensemble of Inception V3, Inception V4, Inception ResNet V2, and ResNet V2 152 with epsilon 16. The success rates against the eight defenses are:

<img src="https://github.com/dongyp13/Translation-Invariant-Attacks/blob/master/results.png">

### Citation
If you use our method for attacks in your research, please consider citing

    @inproceedings{dong2019evading,
      title={Evading Defenses to Transferable Adversarial Examples by Translation-Invariant Attacks},
      author={Dong, Yinpeng and Pang, Tianyu and Su, Hang and Zhu, Jun},
      booktitle={Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition},
      year={2019}
    }

## Implementation

### Models
The models can be downloaded at [Inception V3](http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/inception_v3.ckpt), [Inception V4](http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/inception_v4.ckpt), [Inception ResNet V2](http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/inception_resnet_v2_2016_08_30.ckpt.ckpt), and [ResNet V2 152](http://ml.cs.tsinghua.edu.cn/~yinpeng/downloads/resnet_v2_152.ckpt).

If you want to attack other models, you can replace the model definition part to your own models.

### Hyper-parameters
* For TI-FGSM, set ``num_iter=1``, ``momentum=0.0``, ``prob=0.0``;
* For TI-MI-FGSM, set ``num_iter=10``, ``momentum=1.0``, ``prob=0.0``;
* For TI-DIM, set ``num_iter=10``, ``momentum=1.0``, ``prob=0.7``;
