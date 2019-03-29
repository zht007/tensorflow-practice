前面两篇文章介绍了Tensorflow，以及用Tensorflow快速上手一个线性回归的项目。

实际上Tensorflow对于机器学习新手来说并不是很友好，Tensorflow自己也有高层API，比如Estimator和tf.data就相对来说更容易上手。

Keras本来是独立的机器学习框架，但已经逐渐被整合进了Tensorflow, 今天我们就来简单上手一下吧。

同样的，为了方便与读者交流，所有源代码都放在了
<https://github.com/zht007/tensorflow-practice>

### 1. 创建模拟数据

与[上一篇文章](https://steemit.com/cn-stem/@hongtao/tensorflow)一样，我们还是手动创建一组加了噪音的线性数据。为了模拟真实的数据量，这里的数据点有100万个。

![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-7608cdf848858d3c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
注意，100万个点用plt是画不出来的，图中随机采样了250个点

### 2. Tensorflow方法回顾

与[上一篇文章](https://steemit.com/cn-stem/@hongtao/tensorflow)类似，我们可以直接用Tensorflow来训练这个线性回归的模型，但是需要注意的是，由于数据量非常大，通常我们不会将所有的数据一次性全部丢给机器。而会使用batch的方式，分组喂给机器。

这里我们定义一个batch有10个数据。

```
batch_size = 10
```

在Session 中，我们也采用了随机采样的方式，每个batch从数据中随机抓取10个点。
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-11f7b78d00d4173e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

结果非常不错。

### 2. 训练和验证数据分离

在实际的机器学习项目中，我们需要将原始数据分成训练数据和验证数据(甚至还有单独的测试数据)。

这里我们用到了sklearn中的工具，将数据按照7：3的比例进行了分组。

```
from sklearn.model_selection import train_test_split

x_train, x_eval, y_train, y_eval = train_test_split(x_data,y_true,test_size=0.3)
```

### 3. Keras

#### 3.1 import

Sequential 相当于模型的外壳，Dense是全连接的神经元，SGD是随机梯度下降（Stochastic Gradient Descent）Optimizer.

```
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
```

#### 3.2 创建模型

由于线性回归相当于没有激活函数的神经网络，所以只需要一层Dense就可以了。注意输入输出的shape。

```
model = Sequential()
model.add(Dense(1,input_shape = (1,)))
```

#### 3.3 设置Optimizer

参数就是学习速率

```
sgd = SGD(0.001)
```

#### 3.4 Compile 模型

损失函数compile在了模型中了，'mse'即**均方误差(Mean Square Error MSE)**。

```
model.compile(loss='mse', optimizer=sgd, metrics=['mse'])
```

#### 3.5 训练模型

训练模型超级简单，一句话搞定，循环次数和batch大小也可以定义。

```
model.fit(x_train, y_train, epochs = 1,batch_size = 32)
```

#### 3.6 验证模型

提取最终w和b也非常简单，但是注意数据格式和shape
![img](https://steemitimages.com/640x0/https://cdn.steemitimages.com/DQmbN83Hrb6ks4kkF6vdHVPRcnfF8ZZhFVMTGPh7SbeoHcQ/image.png)
当然，我们也可以用模型来**预测**验证数据。

![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-e2c1555bc5223ae9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 总结

Keras训练模型的过程非常直观，简单来说就三步，第一步，搭建模型；第二步，训练模型；第三步，验证模型。