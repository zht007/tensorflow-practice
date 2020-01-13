![img](https://tva1.sinaimg.cn/large/006tNbRwgy1gavb65g7i0j30rs0ibwfl.jpg)

*Image from [unsplash.com](https://unsplash.com/photos/ZCPwkmfsHNY) by @ripato*

[前面的文章](https://www.jianshu.com/p/e68172ba8c91)我们学习了如何使用 Tensorflow 2.0 训练卷积神经网络，今天我们将学习如何用卷积神经网络的升级版 ResNet 来实战 CIFAR100 数据集。

**关注微信公众号获取源代码（二维码见文末）**

### 1. 深度神经网络的困扰

自从2012 年，5层卷积神经网络在 ILSVRC12 挑战赛 ImageNet 数据集分类取得冠军之后，卷积神经网络再次火爆起来。随着计算机的运算能力的提高，越来约深的神经网络得到了应用。随着神经网络深度的增加，神经网络的准确率也得到了相应的提升。经典网络如 VGG 就将网络深度提高到了19层。

然而神经网络超过20层之后，错误率不仅不会下降，反而上升了，并且训练起来也越来越困难，这主要是由于深度神经网络存在梯度消失和梯度爆炸的问题。在较深层数 的神经网络中，梯度信息由网络的末层逐层传向网络的首层时，传递的过程中会出现梯度 接近于 0 或梯度值非常大的现象。网络层数越深，这种现象可能会越严重。

![VGG13 网络模型结构](https://tva1.sinaimg.cn/large/006tNbRwgy1garzc7yks0j30w006yab6.jpg)

### 2. ResNet 简介

ResNet 的发明就是为了解决深度神经网络梯度消失和梯度爆炸的问题。既然加深网络会导致上诉问题，是否可以给神经网络“短路”，让其有退回稍微浅层结构的能力，至少不能比浅层神经网络差吧。

这种“短路”操作，就是在若干个卷积神经层中间加一条路径，神经网络可以选择

> 1. 经过这几个卷积层完成特征变换。
> 2. 直接跳过这几个卷积层
> 3. 将上面两者结合起来

![添加短路的 VGG13 网络结构](https://tva1.sinaimg.cn/large/006tNbRwgy1garzjunixtj30vm068my8.jpg)

ResNet 由华人科学家何凯明等人在2015年发明，如今已经能够将神经网络扩展到一千多层。感兴趣的读者请前往[这里阅读论文原文](https://arxiv.org/abs/1512.03385)。

### 3. ResNet 结构

论文原文介绍了 18层，34层，50层，101层和152层一共五种ResNet，虽然深度不同，但是结构框架是一致的，只要掌握结构规律，可以将网络结构的层数无限扩散下去。

这里以18层的 ResNet18 为例，介绍一下其基本结构：

最里面的一层是ResNet 最基本的结构，我们这里叫做`BasicBlock`, 该 Block 由两层卷积神经网络外加一个 “短路” 联接构成。若干个 `BasicBlock` 又可以构建一个 `ResBlock`,  4个 `ResBlock` 外加一个预处理层和一个全连接层，一共就构成了 18 层 (2 x 2 x 4  + 2 )的 ResNet18 结构.

![pkPi3](https://tva1.sinaimg.cn/large/006tNbRwgy1gava9ls438j30yq08z0uk.jpg)

### 4. ResNet18 代码实战

首先我们创建最基本单元 `BasicBlcok`

> 这里需要注意以下几点：
>
> 1. BasicBlock 类继承自 tf.keras.layers.Layer 类。
> 2. `num_filters` 缩小图片的尺寸。
> 3. `identity_layer` 必须与卷积神经网络的输出尺寸一致，这里使用 1x1 的filter 来调节。
> 4. 涉及到 batch normaisation 的记住传入training 参数，training在验证的时候需要设置为 False .

```python
class BasicBlock(layers.Layer):
    def __init__(self, num_filters, stride = 1):
        super().__init__()

        self.conv1 = layers.Conv2D(num_filters,(3,3),strides=stride,padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.ReLU()

        self.conv2 = layers.Conv2D(num_filters,(3,3),strides=1,padding='same')
        self.bn2 = layers.BatchNormalization()

        if stride != 1:
            self.identity_layer = layers.Conv2D(num_filters,(1,1),strides=stride)
        else:
            self.identity_layer = lambda x:x

    def call(self, input, training = None):
        output = self.conv1(input)
        output = self.bn1(output, training=training)
        output = self.relu(output)

        output = self.conv2(output)
        output = self.bn2(output,training=training)

        identity_out = self.identity_layer(input)

        output = layers.add([output,identity_out])
        output = tf.nn.relu(output)

        return output
```

然后，用基本单元构建 ResBlock 层，然后再由ResBlock 和 Pre_process 以及最后一个全链接层构建一个 ReNet 模型

> 这里需要注意以下几点:
>
> 1. ResNet class 模型继承 keras.Model.
> 2. Layer_dim 存储 resblock 维度信息
> 3. `layers.GlobalAveragePooling2D()` 不关心图片尺寸，可以在 h * w 纬度上平均尺化。
> 4. 这里模型输出是 logits 没有加 softmax 层

```python
class ResNet(keras.Model):
    def __init__(self, layer_dim, class_num): #layer_dim res18:[2,2,2,2] or res34[3,4,6,3]
        super().__init__()
        self.pre_process = Sequential([layers.Conv2D(64,(3,3),strides = 1, padding = 'same'),
                                       layers.BatchNormalization(),
                                       layers.ReLU(),
                                       layers.MaxPool2D((2,2),strides= 1, padding = 'same')
                                      ])

        self.resblock0 = self.build_resblock(64, layer_dim[0], stride = 1)
        self.resblock1 = self.build_resblock(128, layer_dim[1], stride = 2)
        self.resblock2 = self.build_resblock(256, layer_dim[2], stride = 2)
        self.resblock3 = self.build_resblock(512, layer_dim[3], stride = 2)

        self.avg_pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(class_num)

    def build_resblock(self,filter_num,basic_block_num, stride):

        resblock = Sequential()
        for _ in range(basic_block_num):
            basic_block = BasicBlock(filter_num, stride = stride)
            resblock.add(basic_block)

        return resblock

    def call(self, input, training = None):
        output = self.pre_process(input, training = training) #[b, 64, h, w]

        output = self.resblock0(output, training = training) #[b, 64, h, w]
        output = self.resblock1(output, training = training) #[b, 128, h, w]
        output = self.resblock2(output, training = training) #[b, 256, h, w]
        output = self.resblock3(output, training = training) #[b, 512, h, w]

        output = self.avg_pool(output) #[b, 512, 1,1]
        output = self.dense(output) #[b, class_num]

        return output

```

最后，ResNet18 中间是 [2, 2, 2, 2] 结构，既4个resblock, 每个resblock包涵两个BasicBlcok。 ResNet34 中间为[3, 4, 6, 3]结构。实例ResNet18 和 ResNet34 也非常简单，仅需要传入不同的 layer_dim 即可。

```python
## ResNet 18
model = ResNet([2,2,2,2], 100)

## ResNet 34
model = ResNet([3,4,6,3], 100)
```

模型训练和验证过程请参考[之前文章](https://www.jianshu.com/p/e68172ba8c91)或者查看源代码，就不在这里赘述了。



------

相关文章

[Tensorflow2.0-数据加载和预处理](https://www.jianshu.com/p/b796823ad32c)

[Tensorflow 2.0 快速入门 —— 引入Keras 自定义模型](https://www.jianshu.com/p/e68172ba8c91)

[Tensorflow 2.0 快速入门 —— 自动求导与线性回归](https://www.jianshu.com/p/c44705808f7e)
[Tensorflow 2.0 轻松实现迁移学习](https://www.jianshu.com/p/54aa43935c2b)
[Tensorflow入门——Eager模式像原生Python一样简洁优雅](https://www.jianshu.com/p/883addf4a1b2)
[Tensorflow 2.0 —— 与 Keras 的深度融合](https://www.jianshu.com/p/9a5ae5d92dba)

------

首发[steemit](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2F%40hongtao)

欢迎扫描二维码关注我的微信公众号“tensorflow机器学习”，一起学习，共同进步

![image-20200113142403372](https://tva1.sinaimg.cn/large/006tNbRwgy1gavauvmfmej30do0d8wil.jpg)