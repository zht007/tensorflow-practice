### 1.什么是Tensorflow

Tensorflow是Google 开发的，一个用于机器学习，特别是深度学习的一个Python框架(也支持其他语言)。与Tensorflow类似的框架非常多，但Tensorflow的是目前是最流行的深度学习框架。

### 2.学习Tensorflow之前

总结3条学Tensorflow之前必备技能，就3条

- 一定的python基础，不需要特别精通，但是基本语法要知道。
- 有一定的机器学习，特别是深度学习的知识，不然你学Tensorflow干嘛。
- 会用搜索引擎，勇于阅读官方文档(虽然是Google的，但墙内有官方镜像，真是良心- <https://tensorflow.google.cn/> 不过搜索引擎最好用什么，你懂的)

### 3.Tensorflow原理

这个可能是困扰初学者最多的地方，Tensorflow不像Python中变量赋值、计算、结果可以立即输出。Tensorflow首先要构造计算图谱(graph)，然后再在Session 中计算。 数据(Tensor)在计算图谱中流动(Flow)，就完成了计算，所以这个框架就叫做Tensorflow。

比如在Python中:
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-48062db6cd7ad74e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
c 可以立即输出3，然而在tensorflow中, c只是表示 a+b 的一个计算图谱:
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-dbfcd7b9bed603e8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
一定要在Session() 中运行，就可以得到我们想要的结果：
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-36e4a495314d5088.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
with ... 后面那一堆可以理解成一个固定套路，运行tensorflow必须使用这一步。

### 3.Tensorflow的常见数据类型

#### 3.1 Constant (常量)

这个很好理解，就是固定的数值，定义如下

```
tf.constant()
```

当然，注意当你想定义一个多维矩阵的时候，要先定义一个一维向量，然后再定义其Shape，比如想定义2x2矩阵：
[[1 2]
[3 4]]
需要这样

```
 tf_const = tf.constant([1,2,3,4],shape=(2,2))
```

当然tensorflow中有非常有用的，用于快速定义：全0的，全1的，随机正态分布(random normal)的，以及随机均匀分布(random uniform)的Tensor

```
my_ones = tf.ones((2,2))
my_zeros = tf.zeros((2,2))
my_rand = tf.random_normal((2,2))
my_randuni = tf.random_uniform((2,2),minval=10,maxval=20)
```

#### 3.2 Variable(变量)

Tensorflow的变量的定义和使用都比较特殊。

- 首先，变量需要定义一个初始值，初始值可以是之前定义tf.constant，也可以用tf.ones 和 tf.random_normal等生成。

```
a = tf.Variable(tf_const)
b = tf.Variable(tf.ones((2,2)))
```

- 其次，计算variable的时候需要先初始化，这里要比constant多一个步骤

```
init = tf.global_variables_initializer()
```

*最后在Session中记得run这个init就可以了

```
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(a+b))
```

可以看到输出的数值就是a+b的值了
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-b1e78a5d08f1df3d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

和Python中的变量一样，变量的数值是可以改变的，比如，我们先让a=a+b，再进行a+b的运算，就可以得到
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-254023cc5c32ee9f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 3.3 Placeholder

Placeholder在Tensorflow中有特殊用途，顾名思义就是一个占位变量，在计算图谱中先占一个位置，在run session的时候再通过feed_dict赋值。
定义Placeholder非常简单，可以不用设置初始值，但一定要指定数据类型。

```
c = tf.placeholder(dtype=tf.float32)
```

当然我们可以定义一个python变量以方便在计算的时候赋值，feed_dic是支持phtyon和numpy的变量，但是不支持tensorflow的变量，这里需要注意。

```
d = [1,2,3,4]
```

最后在Session中我们通过feed_dic赋值并计算就可以啦
![img](https://steemitimages.com/640x0/https://upload-images.jianshu.io/upload_images/10816620-49e995c3504251fc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 4.尾巴

知道以上三条，就可以用Tensorflow来做矩阵计算啦，是不是特别简单。

------

同步到我的简书和Steemit
<https://www.jianshu.com/u/bd506afc6fc1>

<https://steemit.com/@hongtao>

