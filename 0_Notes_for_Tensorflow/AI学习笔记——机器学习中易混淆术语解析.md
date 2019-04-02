![blue and orange abstract painting](https://images.unsplash.com/photo-1553949345-eb786bb3f7ba?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

*image source from [unsplash.com](https://images.unsplash.com/photo-1553949345-eb786bb3f7ba?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1950&q=80) by [Paweł Czerwiński](https://unsplash.com/@pawel_czerwinski)*

对于机器学习的初学者来说，容易被一些专用术语搞得晕头转向，这篇文章我们就从挑几个容易混淆的概念和术语接解析一下。

### 1. Batch

在机器学习的过程中，有时候由于样本数量巨大，无法一次性送入模型进行学习，需要分成若干**Batch**(批次)，将样本分批(**Batch**)送入模型的学习叫做**Batch Learning**。一个Batch中样本的数量叫做**Batch Size**。

> Number of Batches = Number of Samples / Batch Size 

当然Batch Learning 不一定要遍历所有样本，Random Batch 就是随机从样本中取batch size数量的样本送入模型中进行学习，就有可能出现样本被重复选择的情况。

### 2. Iteration(Step)

Iteration, Step 和 Epoch 三个概念最容易混淆，Iteration 和 Step 概念是一致的，表示完成一次学习过程。

对于Batch Learning, 学完**一个**Batch的过程就是完成一个Iteration 或者 Step。

### 3. Epoch

一个Epoch是指所有样本都完成了一次训练。对于Batch Learning 来说，需要将样本中所有Batch都学完。

> One Epoch = Number of Iterations (Number = Number of Batches)

---

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