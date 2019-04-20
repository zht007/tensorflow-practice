![photo of milk bottle lot](https://ws3.sinaimg.cn/large/006tNc79gy1g24pu1tcibj30rs0rs0vg.jpg)

*image source from [unsplash](https://images.unsplash.com/photo-1523473827533-2a64d0d36748?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1100&q=80) by Mehrshad Rajabi*

[上一篇文章](https://steemit.com/cn-stem/@hongtao/tensorflow-rnn)我们用Keras搭建GRU神经网络，通过对前13年牛奶产量的学习，成功预测了地最后1年牛奶的产量。

该模型是**多对一**的输入/输出结构，也就意味着12个月的数据输入，只能输出1个月的数据。有没有可能改进模型，让输出输入的数量一致，以提高预测效率呢？这篇文章我们就来改进GRU模型，实现**多对多**的结构。

同样的，为了方便与读者交流，所有的代码都放在了这里：

#### Repository:

https://github.com/zht007/tensorflow-practice

### 1. 数据预处理

数据的导入，**训练集测试集分离**以及**归一化**与之前一致，就不赘述了。需要改变的是GRU输入输出Shape。

> - 设计一个连续的**数据窗口**，窗口中包含24个月的数据。
> - 由于是**多对多**的结构，前12个月数据X为输入的Feature，后12个月数据为label与神经网络的输出做对比。
> - **数据窗口**按月平移，这样一共可以产生13*12个组数据。

采用相同的帮助函数，仅仅改变future_monthes的数量

```python
def build_train_data(data, past_monthes = 12, future_monthes = 12):
  X_train, Y_train = [],[]
  
  for i in range(data.shape[0] - past_monthes - future_monthes):
    X_train.append(np.array(data[i:i + past_monthes]))
    Y_train.append(np.array(data[i + past_monthes:i + past_monthes + future_monthes]))
    
  return np.array(X_train).reshape([-1,12]), np.array(Y_train).reshape([-1,12])
     
```

调用帮助函数获得输入和输出

```python
x, y = build_train_data(train_scaled)
```

### 2. GRU神经网络

#### 2.1 None-Stateful结构 

**多对多**结构的GRU与**多对一**的GRU结构没有太大的变化，唯一的区别是最后的Dense层需要用 layers.TimeDistributed()连接，以便将所有时间序列上的输出都传给Dense层，而不仅仅是最后一位。

```python
model_layers = [
    layers.Reshape((SEQLEN,1),input_shape=(SEQLEN,)),
    layers.GRU(RNN_CELLSIZE, return_sequences=True),
    layers.GRU(RNN_CELLSIZE, return_sequences=True),
    layers.TimeDistributed(layers.Dense(1)),
    layers.Flatten()
    
]
model = Sequential(model_layers)
model.summary()
```

*部分代码参考 [github](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rnn-tutorial/00_Keras_RNN_predictions_playground.ipynb) with lisence Apache-2.0*

GRU的结构如下图所示

![image-20190416121948606](https://ws2.sinaimg.cn/large/006tNc79gy1g24ozcm9nmj30vm0g8tan.jpg)

训练1000个epoch后的loss变化如图：

![image-20190416122309696](https://ws2.sinaimg.cn/large/006tNc79gy1g24p2ts2p4j30my0hsgol.jpg)

我们发现loss在下降的过程中**噪音非常大**，而且这反复的变化似乎是成规律的。这是由于我们在训练的过程中，每个Batch是相对独立的，其训练之后产生的**状态(State)**并没有传到下一个Batch中。要解决这个问题，我们需要在GRU中开启Stateful。

#### 2.2 Stateful结构

开启Stateful之后，我们必须在第一个输入层指定batch_size，为了方便后面的预测，这里的batch_size 设定为1。记得GRU的每一层都需要开启Stateful。

```python
RNN_CELLSIZE = 10
SEQLEN = 12
BATCHSIZE = 1

model_layers = [
    layers.Reshape((SEQLEN,1),input_shape=(SEQLEN,),batch_size = BATCHSIZE),
    layers.GRU(RNN_CELLSIZE, return_sequences=True, stateful=True),
    layers.GRU(RNN_CELLSIZE, return_sequences=True, stateful=True),
    layers.TimeDistributed(layers.Dense(1)),
    layers.Flatten()
    
]
model = Sequential(model_layers)
model.summary()
```

*部分代码参考 [github](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd/blob/master/tensorflow-rnn-tutorial/00_Keras_RNN_predictions_playground.ipynb) with lisence Apache-2.0*

如下图所示，这里我们仅仅训练了100个epoch，loss的下降就非常平滑了。

![image-20190416123012594](https://ws1.sinaimg.cn/large/006tNc79gy1g24pa5u1pzj30nc0hgwfw.jpg)

### 3. 模型预测

同样的，我们用模型预测最后一年12个月牛奶的产量，这里我们将三个模型的预测结果做了对比。

![image-20190416124233619](https://ws2.sinaimg.cn/large/006tNc79gy1g24pmzs90ij30oo06ldh1.jpg)

可以看到**多对多**的模型，尤其是开启了Stateful之后的预测更加地准确。

![image-20190416124158893](https://ws3.sinaimg.cn/large/006tNc79gy1g24pmewkz1j30ot06y0ut.jpg)

用第一年的数据生成后面13年的数据如上图所示，可以发现，**多对多**的模型同样更具优势。

-----------

参考资料

[1]<https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0>

[2]<https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git>

[3]<https://www.tensorflow.org/api_docs/>

[3]<https://datamarket.com/data/set/22ox/monthly-milk-production-pounds-per-cow-jan-62-dec-75#!ds=22ox&display=line>

------

相关文章

[Tensorflow入门——RNN预测牛奶产量](https://steemit.com/cn-stem/@hongtao/tensorflow-rnn)

[AI学习笔记——循环神经网络（RNN）的基本概念](https://steemit.com/rnn/@hongtao/ai-rnn)

[Tensorflow入门——单层神经网络识别MNIST手写数字](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)

[Tensorflow入门——多层神经网络MNIST手写数字识别](https://steemit.com/cn-stem/@hongtao/6qe2nw-tensorflow-mnist)

[AI学习笔记——Tensorflow中的Optimizer](https://steemit.com/tensorflow/@hongtao/ai-tensorflow-optimizer)

[Tensorflow入门——分类问题cross_entropy的选择](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)

[AI学习笔记——Tensorflow入门](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)

[Tensorflow入门——Keras简介和上手](https://steemit.com/cn-stem/@hongtao/tensorflow-keras)

------

同步到我的简书

<https://www.jianshu.com/u/bd506afc6fc1>