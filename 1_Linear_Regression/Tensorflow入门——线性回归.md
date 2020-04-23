[上一篇文章](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)我们介绍了Tensorflow的基础，趁热打铁，这篇文章介绍一下如何用Tensorflow训练一个线性回归的项目。

为了方便读者交流和学习，我在github上创建了一个repo，地址在这里
<https://github.com/zht007/tensorflow-practice>

#### 1. import

首先当然是要import tensorflow，numpy 以及以及数据可视化工具

```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

这里"%matplotlib inline" 用在jupyter notebook，可以使数据可视化交互更加友好。

#### 2. 创建模拟数据

这里我们要创建一个线性回归的模拟数据，y = x，一共20个点，再在这些点上加一些随机的噪音，数据用numpy生成。

```
x_data = np.linspace(0,10,20) + np.random.uniform(-1.5,1.5,20)
y_data = np.linspace(0,10,20) + np.random.uniform(-1.5,1.5,20)
```

![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-961a05d058277d65.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 3. 初始化w 和 b

线性回归中，训练的参数就是权重(weight)w 和偏移(bais)b，我们这里用numpy随机生成w和b的初始值。

当然w和b是tensorflow中训练的对象，所以记得转换成tensorflow的变量。
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-b99c78d65019b407.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 4. 定义(error)损失函数

训练的目标是降低error，我们需要在这里手动定义error是什么，这里我定义的是：**均方误差(Mean Square Error MSE)**。

```
error = 0
for x,y in zip(x_data,y_data):
    error += (w_tf * x + b_tf - y)**2
```

#### 5. 定义Optimizer(优化器)

线性回归，需要通过Gradient Decendent(梯度下降方法)训练w, 和 b。Tensorflow中提供了这个Optimizer。

同时不要忘了，定义train 来最小化 error.

```
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
```

#### 6. 开始训练啦

这里就是Tensorflow的标准流程啦，不要忘了初始化，以及将最后训练的结果w_final 和 b_final 提取出来。

注意这里我们定义了训练的次数train_steps

```python
init = tf.global_variables_initializer()

train_steps = 10
with tf.Session() as sess:
    sess.run(init)
    for step in range(train_steps):
        sess.run(train)
    w_final, b_final = sess.run([w_tf,b_tf])
```

#### 7. 检验训练成果

最后，我们把原始数据和训练得到的结果通过plt.plot可视化，就一目了然啦。
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-886100d159472ffe.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 尾巴

本文通过简单几步就完成了Tensorflow的线性回归项目的创建和训练，是不是很简单。

------

同步到我的简书和Steemit
<https://www.jianshu.com/u/bd506afc6fc1>

<https://steemit.com/@hongtao>