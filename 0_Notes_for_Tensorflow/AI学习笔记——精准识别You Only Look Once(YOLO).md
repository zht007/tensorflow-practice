[上一篇文章](https://steemit.com/cn-stem/@hongtao/2jkjay-tensorflow-mnist)我们讲到，通过**卷积神经网络(CNN)**可以轻松实现图片的识别与分类。所以，只要有足够多的，已标记的样本，我们就可以用这些样本将神经网络训练成分类器，用来识别新的图片。MINST手写数字的样本，可以训练手写数字分类器；不同品种猫的样本，可以训练出猫的分类器；人脸的样本可以训练出人脸分类器...

对于人类来说，给我们一张图片，或者我们瞄一眼一个画面，画面或图片中的每个物品和物品的相应位置就能立即被我们识别。然而，上面提到分类器似乎只能识别单一物品，即便能识别多个物品，也无法指出物品在图片中的位置。

这篇文章我们就来聊聊**YOLO**[[1]](https://arxiv.org/abs/1506.02640)这个神奇的模型，它可以像人一样，一眼就能识别图片中的物体以及物体的位置。更加神奇的是，对于动态视频甚至摄像头实时采集的画面，YOLO都能精准识别。

![](https://ws4.sinaimg.cn/large/006tNc79gy1g1w0ktfex4g30go09e7wu.gif)

*image source from [github](https://github.com/thtrieu/darkflow) with GNU General Public License v3.0*



## 1. 目标识别简史

2001年左右，Viola-Jones的人脸识别算法[[2]](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework#cite_note-1)是当时准确度较高的目标识别算法，这个算法需要先标记出**人脸的特征值**，比如眼睛鼻子间的距离等，然后再送给机器使用支撑向量(SVM)的方法进行训练。

2005年，N. Dalal[[3]](https://hal.inria.fr/inria-00548512/)将人脸的图片先进行灰度分析，形成**灰度特征向量(Histograms of Oriented Gradient)**，再将这些预处理过的图片送入计算机进行分类训练，大大提高了人脸识别的精确度。

2012年， Kriszhevsky团队用**卷积神经网络(CNN)**将人脸识别的准确率再提高了一个台阶，从此目标识别领域迈进了*深度学习*的时代。

正如本文开篇讲到，普通的CNN网络只能对单个物体进行识别，而且无法指出物体的位置。为了解决这个问题，**[R-CNN](https://arxiv.org/abs/1311.2524)**算法就诞生了。R-CNN需要对图片扫描多次，先框出图片可能有物体的区域，然后再对这些区域用CNN算法进行识别。

## 2. YOLO原理简介

**YOLO—You Only Look Once**，顾名思义，就是对图片仅进行一次扫描就能同时识别多个物体，并标记出物体的位置。YOLO目前已经迭代出了多个版本，但是基本原理没有改变，我们以YOLO V2为例。

* 首先，YOLO将一张图片分成13×13个**格子**。

![Alt Text](https://camo.githubusercontent.com/4338301d905b87a1e9d8a4b68b63775d30adcf15/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f477269644032782e706e67)

*image sorce from [github](https://github.com/llSourcell/YOLO_Object_Detection/blob/master/YOLO%20Object%20Detection.ipynb) with GNU General Public License v3.0*

* 然后，YOLO对每个格子输出5个**预测框**。预测框的粗细可以表明这个框内存在目标物体的可能性。

![Alt Text](https://camo.githubusercontent.com/c5b0bb9269257ba704824dedd8bc03b62d40ed61/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f426f7865734032782e706e67)

*image sorce from [github](https://github.com/llSourcell/YOLO_Object_Detection/blob/master/YOLO%20Object%20Detection.ipynb) with GNU General Public License v3.0*

* 其次，这些预测框同时可以预测目标物体的类别，如下图，用不同颜色标记不同**类别(20个)**，粗细同样表示可能性的大小。

![Alt Text](https://camo.githubusercontent.com/d1becdbc2064341b828ae5a73d6f92f4a8a27308/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f53636f7265734032782e706e67)

*image sorce from [github](https://github.com/llSourcell/YOLO_Object_Detection/blob/master/YOLO%20Object%20Detection.ipynb) with GNU General Public License v3.0*

* 最后，YOLO将可能性比较大(比较粗)的预测框输出，就得到我们想要的结果了。

![687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f50726564696374696f6e4032782e706e67](/Users/hongtao/Downloads/687474703a2f2f6d616368696e657468696e6b2e6e65742f696d616765732f796f6c6f2f50726564696374696f6e4032782e706e67.png)

*image sorce from [github](https://github.com/llSourcell/YOLO_Object_Detection/blob/master/YOLO%20Object%20Detection.ipynb) with GNU General Public License v3.0*

## 3. YOLO结构

如下图，所示，YOLO是由我们熟悉的若干个CNN堆叠而成的，我们来看看原始图片经过多层CNN处理之后，最后一层输出的是什么。

![Alt Text](https://camo.githubusercontent.com/3c2151338f97e8494cb208d46a29bab4763c7dd6/68747470733a2f2f692e696d6775722e636f6d2f5148304376524e2e706e67)

YOLO最后一层输出的是一个13×13×125的矩阵，所有我们需要的信息都包含在了这一层中。

> * 13×13就是之前提到的图片中被划分成的13×13个**格子**。
>
> * 每个格子包含125位信息，这125位信息包含5个**预测框**，每个预测框含有25位信息(125= 5×25)。
>
> * 这25位信息，前五位为**[x, y, width, height, Score]**；其中，(x, y) 表示预测框的**中心坐标**，width和height表示预测框的**长，宽**，Score表示预测框中有目标的**可能性**。
>
> * 这25位信息，后20位每一位代表一个**类别**，其数表示属于该类别的概率。

## 4. 总结

高大上的YOLO经过一步步分解之后，原理是不是并不复杂。当然本文只是对YOLO的粗浅介绍，想要深入了解YOLO请见文末参考文献和[YOLO官网](https://pjreddie.com/darknet/yolo/)。下一篇文章，我将介绍如何使用YOLO来检测和识别自己的图片和视频中的目标物体。

----

## 参考资料

[1] [J. Redmon et al. You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

[2] [Viola–Jones object detection framework](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework#cite_note-1)

[3] [N. Dalal et al. Histograms of Oriented Gradients for Human Detection](https://hal.inria.fr/inria-00548512/)

[4] [R. Girshick et al. Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)

[5] [YOLO official website.](https://pjreddie.com/darknet/yolo/)

------

相关文章

[Tensorflow入门——单层神经网络识别MNIST手写数字](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)

[Tensorflow入门——多层神经网络MNIST手写数字识别](https://steemit.com/cn-stem/@hongtao/6qe2nw-tensorflow-mnist)

[AI学习笔记——Tensorflow中的Optimizer](https://steemit.com/tensorflow/@hongtao/ai-tensorflow-optimizer)

[Tensorflow入门——分类问题cross_entropy的选择](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)

[AI学习笔记——Tensorflow入门](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)

[Tensorflow入门——Keras简介和上手](https://steemit.com/cn-stem/@hongtao/tensorflow-keras)

------

同步到我的简书

<https://www.jianshu.com/u/bd506afc6fc1>
