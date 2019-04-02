![img](https://images.unsplash.com/photo-1553949312-1b37019c1c15?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

*Image source: [unsplash.com](https://images.unsplash.com/photo-1553949312-1b37019c1c15?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1350&q=80) by Paweł Czerwiński*

[之前的文章](https://steemit.com/cn-stem/@hongtao/6qe2nw-tensorflow-mnist)我们介绍了如何用单层和多的**全连接**层神经网络识别手写数字，虽然识别率能够达到98%，但是由于全链接神经网络本身的局限性，其识别率已经很难再往上提升了。我们需要改进神经网络的结构，采用**卷积神经网络(CNN)**的结构来进一步提高的识别率。

关于CNN的原理，我在[之前的文章](https://steemit.com/cn/@hongtao/ai-1-cnn)中已经介绍，这篇文章就不过多赘述，我们直接进入实战阶段。

同样的，为了方便与读者交流，所有的代码都放在了这里：

#### Repository:

https://github.com/zht007/tensorflow-practice



## 1. 初始化W和B

卷积神经网络中权重W的shape尤其重要，CNN中的W实际上就是一个四维的filter，这个四维的filter由n个三维filter堆叠而成，n的大小等于输出channel的深度。当然三维filter又是由m个二维filter堆叠的，m的大小等于输入Channel的深度。

动画效果可以参见[这里](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#10)。

> **W的shape为[filter[0], filter[1], input_channel_depth, output_channel_depth]**
>
> 例如W[6,6,3,4] 表示：二维的filter的size是(6,6), 输入的图片有3个Channel, 输出的图片有4个Channel

偏置B的Shape与output_channel保持一致即可，tensorflow会自动broadcast成正确的维度，B在这里与多层神经网络的的初始化相同。

神经网络的结构一共5层，3层CNN，2层全链接，最后一层与单层神经网络一样，10个神经元输出识别结果：数字是是0-9的概率。

```python
# three convolutional layers with their channel counts, and a
# fully connected layer (the last layer has 10 softmax neurons)
K = 12  # first convolutional layer output depth
L = 24  # second convolutional layer output depth
M = 48  # third convolutional layer
N = 200  # fully connected layer

W1 = tf.Variable(tf.truncated_normal([6,6,1,K], stddev=0.1)) 
B1 = tf.Variable(tf.ones([K])/10)

W2 = tf.Variable(tf.truncated_normal([5,5,K,L], stddev=0.1))
B2 = tf.Variable(tf.ones([L])/10)

W3 = tf.Variable(tf.truncated_normal([4,4,L,M], stddev=0.1))
B3 = tf.Variable(tf.ones([M])/10)

W4 = tf.Variable(tf.truncated_normal([7*7*M,N], stddev=0.1))
B4 = tf.Variable(tf.ones([N])/10)

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))
```

*该部分代码部分参考[[2]](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)[[3]](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git) with Apache License 2.0*

## 2. 神经网络搭建

CNN的部分，我们用tensorflow自带的[tf.nn.conv2d()](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)方法：

```python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=True,
    data_format='NHWC',
    dilations=[1, 1, 1, 1],
    name=None
)
```

用Tensorflow搭建神经网络的时候注意以下几点:

>1. Padding 这里使用的是'SAME'，也就是步长(stride)为1的时候输入与输出图片的shape保持一致。
>2. 这里没有使用Max-Pooling层来"压缩"图片，而是增加stride(第二层和第三层Stride 为2)的方式，效果是一样的。28x28的图片经过两层CNN之后，压缩成了14x14和7x7的图片。
>3. CNN与全连接神经网络连接之前，需要将CNN输出的图片拆开拼接成一维的向量(Flatten or Reshape)。

```python
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding='SAME') + B1)

Y2 = tf.nn.relu(tf.nn.conv2d(Y1,W2, strides = [1,2,2,1], padding='SAME') + B2)

Y3 = tf.nn.relu(tf.nn.conv2d(Y2,W3, strides = [1,2,2,1], padding='SAME') + B3)

#flat the inputs for the fully connected nn
YY3 = tf.reshape(Y3, shape = (-1,7*7*M))
                

Y4 = tf.nn.relu(tf.matmul(YY3, W4) + B4)
Y4d = tf.nn.dropout(Y4,rate = drop_rate)

Ylogits = tf.matmul(Y4d, W5) + B5
Y = tf.nn.softmax(Ylogits)
```

*该部分代码部分参考[[2]](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)[[3]](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git) with Apache License 2.0*

## 3. 识别效果

在其他参数都没改变的情况下，仅仅改变了神经网络的结构，可以看出识别率已经超出99%了。

![image-20190402143453530](https://ws3.sinaimg.cn/large/006tKfTcgy1g1om7o32cej30b5098mxj.jpg)

目前我通过CNN的神经网络训练出来的分类器参加Kaggle的比赛，最好成绩是识别率99.3，全球排名第792名。

![image-20190402144054498](https://ws4.sinaimg.cn/large/006tKfTcgy1g1omdw7vj6j30u20anwg0.jpg)

## 4. CNN结构的Keras实现

如果用Keras这个高级的API搭建CNN就更加简单了，无需初始化W和B，只需要关心神经网络的结构本身就行了。

使用Keras的[layers.Conv2D()](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)方法，注意其中的参数filters 是输出Channel的depth，Kernel_size 是二维filter的shape，实现相同结构的代码如下:

```python
model = models.Sequential()

model.add(layers.Conv2D(filters = 12, kernel_size=(6,6), strides=(1,1),
                       padding = 'same', activation = 'relu',
                       input_shape = (28,28,1)))
          
model.add(layers.Conv2D(filters = 24,kernel_size=(5,5),strides=(2,2),
                       padding = 'same', activation = 'relu'))

model.add(layers.Conv2D(filters = 48,kernel_size=(4,4),strides=(2,2),
                       padding = 'same', activation = 'relu'))
          
model.add(layers.Flatten())          
          
model.add(layers.Dense(units=200, activation='relu'))
model.add(layers.Dropout(0.25))

model.add(layers.Dense(units=10, activation='softmax'))
```





------

参考资料

[1][https://www.kaggle.com/c/digit-recognizer/data](https://www.kaggle.com/c/digit-recognizer/data)

[2][https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)

[3][https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git)

[4][https://www.tensorflow.org/api_docs/](https://www.tensorflow.org/api_docs/)

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

