Tensorboard 是 Tensorflow 中的可视化工具，使用 Tensorboard 不仅可以查看计算图谱(神经网络)结构，而且还能够将训练过程中参数变化，准确率以及损失函数的变化，直观地展示出来。是机器学习研究者非常有用的工具，这篇文章，我就来介绍一下如何在 Keras 中轻松调用Tensorboard。

同样的，为了方便与读者交流，所有的代码都放在了这里：

https://github.com/zht007/tensorflow-practice

### 1. 导入TensorBoard

Tensorboard 在 Keras 中是以 callback 的形式调用的。

```python
from tensorflow.keras.callbacks import TensorBoard
```

然后我们需要初始化 tensorboard。这里为了避免命名重复，在文件名中加入了时间参数。

```python
import time
NAME = 'DigiRecognizer-CNN-{}'.format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))
```

注意，这里我们定义了一个"logs/"的目录。

### 2. TensorBoard使用

使用Tensorboard 非常简单，我们以之前识别手写数字的 MINST 项目为例，在保持所有其他代码不变的情况下，仅需要在 model.fit 的函数中加入callbacks的参数即可。不过注意的是 callback 需要一个List 所以，这里直接 callbacks=[tensorboard] 就可以了

```python
model.fit(x_train, y_train,
          batch_size=50,
          epochs=100,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

### 3. 显示TensorBoard

Tensorboard 的显示需要用到浏览器，这里最好使用Google 的 Chrome浏览器。使用Terminal 或者 cmd , cd 到工作目录下，使用这个命令

```python
tensorboard --logdir logs
```

即可获得一个url，复制粘贴这个url到浏览器即可看到Tensorboard啦。

###4. 在Colab中使用TensorBoard

我们通常无法直接在Jupyter notebook 或者 Colab 中显示 TensorBoard，不过Tensorflow 2.0 已经支持这个功能了，也仅需几行代码。

在Colab中开一个代码格，即可安装Tensorflow 2.0 alpha

```python
%%capture
!pip install -q tensorflow-gpu==2.0.0-alpha0
# Load the TensorBoard extension
%load_ext tensorboard.notebook
```

其他部分不用改变，训练完成后，仅需一行代码，即可在Colab的Notebook中直接显示TensorBoard 

```python
%tensorboard --logdir logs
```



Tensorboard 数据展示非常美观，下图为在全连接的神经网络中改变网络结构，训练了三次，所得到的准确率和损失函数，在训练集和测试集的结果对比。通过Dropout和正则化能够改善过拟合的现象(灰色和橙色分别是改善后的训练集和测试集结果)，但是还是无法进一步提高该神经网络的识别率了。

![006tNc79gy1g3ytlhvkq4j30be0hn76e](http://ww4.sinaimg.cn/large/006tNc79gy1g40tnlj510j30be0hnt9x.jpg)

因此，下一步就必须改变神经网络的结构，引入卷积神经网络来进一步提高模型的识别率，同时避免Overfitting。下一篇文章我会具体介绍如何使用 Tensorboard 工具调整卷积层层数，每层神经网络个数等参数，最后达到最优神经网络的目的。

------

相关文章

[Tensorflow入门——单层神经网络识别MNIST手写数字](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)

[AI学习笔记——Tensorflow中的Optimizer](https://steemit.com/tensorflow/@hongtao/ai-tensorflow-optimizer)

[Tensorflow入门——分类问题cross_entropy的选择](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)

[AI学习笔记——Tensorflow入门](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)

[Tensorflow入门——Keras简介和上手](https://steemit.com/cn-stem/@hongtao/tensorflow-keras)

------

同步到我的简书和Steemit

[https://www.jianshu.com/u/bd506afc6fc1](https://www.jianshu.com/u/bd506afc6fc1)

<https://steemit.com/@hongtao>