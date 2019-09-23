随着 Tensorflow 2.0 正式版发布的日期越来越近，我们应该做好准备忘记 1.0 版本中那些反人类的 静态 Graph 和 Session，拥抱新版本的各种易用简单的新特性。

我们之前的文章介绍了 Tensorflow 的 Eager 模式，Tensorflow 2.0 默认就是在 Eager 模式下运行的，所以 Eager 模式下可以直接打印出运算结果以及与 numpy 的无缝切换等特性，在本文中就不在赘述了，感兴趣的读者可以自行参考前文。

该部分的全部代码请访问我的 github repo 获取

www.github.com/zht007/tensorflow-practice



### 1. Tensorflow 2.0 初始化

写作本文的时候 Tensorflow 2.0 已经出到 RC 版本了，与正式版应该相差不大了。如何安装该版本请参考[官方说明。](https://tensorflow.google.cn/install) 

```python
# upgrade pip
pip install --upgrade pip

# Current stable release for CPU-only
pip install tensorflow

# Preview nightly build for CPU-only (unstable)
pip install tf-nightly

# Install TensorFlow 2.0 RC
pip install tensorflow==2.0.0-rc1
```

如果使用 Colab 只需要在初始的代码框运行这几行代码即可

```python
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
```

### 2. 自动求导

微分求导是机器学习的基础，Tensorflow 提供了强大的自动求导工具，下面我们就用 Tensorflow 2.0 计算函数 y(x)=x^2 在 x=3 时的导数

```python
import tensorflow as tf

x = tf.Variable(initial_value=3.)
with tf.GradientTape() as tape:     # 在 tf.GradientTape() 的上下文内，所有计算步骤都会被记录以用于求导
    y = tf.square(x)
y_grad = tape.gradient(y, x)        # 计算y关于x的导数
print([y.numpy(), y_grad.numpy()])
```

*该部分代码参考 [github repo]( https://github.com/snowkylin/tensorflow-handbook) with MIT License*

我们可以得到 y = 3*3 = 9, y_grad = 2 * 3 =6 的结果。 

> `tf.GradientTape()` 是一个自动求导的记录器，在其中的变量和计算步骤都会被自动记录。在上面的示例中，变量 `x` 和计算步骤 `y = tf.square(x)` 被自动记录，因此可以通过 `y_grad = tape.gradient(y, x)` 求张量 `y` 对变量 `x` 的导数。

### 3. 线性回归

该部分模拟生成的数据，我们之前用 Keras， Tensorflow 1.0 以及 Tensorflow Eager 模式都训练过了，感兴趣的读者可以找来对比一下。

#### 3.1 创建模拟数据

与之前的数据一样，此处数据是100万个带噪音的线性数据，100万个点用plt是画不出来的，图中随机采样了250个点

![image](https://upload-images.jianshu.io/upload_images/10816620-838d35aff75c9e70.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

我们定义一个帮助函数方便以batch的形式这100万个数据点中随机抽取batch size大小的数据进行训练

```python
def next_batch(x_data, batch_size):
    batch_index = np.random.randint(len(x_data),size=(BATCH_SIZE))
    x_train = x_data[batch_index]
    y_train = y_true[batch_index]
    return x_train, y_train
```

#### 3.2 定义变量

此处与 Tensorflow  1.0 变量没有任何区别

```python
w = tf.Variable(initial_value=0.)
b = tf.Variable(initial_value=0.)
```

#### 3.3 定义优化器

为了让 Tensorflow 自动更新参数，我们需要用到优化器，这里我用 SGD 来更新参数。

```python
optimizer = tf.keras.optimizers.SGD(learning_rate= 1e-3)
```

#### 3.4 线性回归模型训练

在这里我们需要理解一下几点

> * 要更新参数就需要对参数求导，这里需要更新的参数(变量)为 w 和 b。
> * 要让模型最优，及需要求变量(w, b)对损失函数求导(偏微分)
> * 与前面一样，使用 `tape.gradient(ys, xs)` 自动计算梯度
> * 使用 `optimizer.apply_gradients(grads_and_vars)` 自动更新模型参数。

```python
for step in range(BATCHS):
  x_train, y_train = next_batch(x_data, BATCH_SIZE)
  with tf.GradientTape() as tape:
    y_pred = w * x_train + b
    loss = tf.reduce_sum(tf.square(y_pred - y_train))/(BATCH_SIZE)
  
  grads = tape.gradient(loss, [w, b])
  optimizer.apply_gradients(grads_and_vars=zip(grads,[w,b]))
```

#### 3.5 训练结果

此处结果也相当符合预期

![image-20190502150808881](https://upload-images.jianshu.io/upload_images/10816620-7eae0c67470ff7dc.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

### 4. 总结

Tensorflow 2.0 提供了一个非常优雅和简洁的方式对参数进行自动求导和自动更新。 `tf.GradientTape()` 记录计算步骤用于自动求导和自动更新参数的方式请务必牢记，在更复杂的模型下也将会被用到。

------

相关文章

[Tensorflow 2.0 轻松实现迁移学习](https://steemit.com/cn-stem/@hongtao/tensorflow-2-0)

[Tensorflow入门——Eager模式像原生Python一样简洁优雅](https://steemit.com/cn-stem/@hongtao/tensorflow-eager-python)

------

同步到我的简书
https://www.jianshu.com/u/bd506afc6fc1