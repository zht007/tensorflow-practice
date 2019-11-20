上篇文章我们用[线性回归的项目快速入门了 Tensorflow 2.0](https://steemit.com/cn-stem/@hongtao/55x5e5-tensorflow-2-0) ，本文我们继续深入研究 ，使用 Keras 搭建模型配合 Tensorflow 2.0 解决线性回归和分类问题。

全部代码，请见我的github repo 

https://github.com/zht007/tensorflow-practice

### 1. 线性回归回顾

在 Tensorflow 2.0 中我们使用`tf.GradientTape()`记录前向传播计算(Forward Propagation)，然后利用 `tape.gradient()`计算梯度，最后使用 `optimizer.apply_gradient()`自动后向传播计算(Backward Propagation) 完成参数更新。

在线性回归的项目中，我们 *手动写出* 来前向计算的公式

```python
y_pred = w * x_train + b
```

在线性回归这样的简单问题，中我们当然可以手写公式，但是如果遇到比较复杂的模型，比如多层神经网络，手写起来就比较麻烦了。尤其是每层参数的 Shape 很容易出错。这个时候，我们可以引入 Keras 的预定义层（全连接，CNN卷积，RNN 等）。

### 2. Keras 模型与层的引入

实际上，我们之前已经用 Keras 很长时间了，只不过之前我们都是用的 `tf.keras.Sequential`的模型堆叠出我们想要的模型，然后 compile 并 fit 训练。这种方法虽然快速简单，但是缺乏一定的灵活性。

> 如何更自由地利用 Keras 预定义层创建更加复杂的神经网络，我们就需要用到 python 面向对象的编程方法，继承`tf.keras.Model` 类并创建自己的 Model。其结构如下

```python
class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
        # layer1 = tf.keras.layers.BuiltInLayer(...)
        # layer2 = MyCustomLayer(...)

    def call(self, input):
        # 此处添加模型调用的代码（处理输入并返回输出），例如
        # x = layer1(input)
        # output = layer2(x)
        return output

    # 还可以添加自定义的方法
```

*code from [github repo](https://github.com/snowkylin/tensorflow-handbook) with MIT lisence*

对于线性回归问题，实际上就是建立一个没有激活函数的，只有一个神经元的全连接神经网络，代码如下

```python
class LinearModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    self.dense = tf.keras.layers.Dense(
        units=1)

  def call(self,input):
    output = self.dense(input)
    return output
```

在训练模型的时候，首先实例化模型`modle = LinearModel()` 然后用 `y_pred = model(x_train)`来代替我们之前手写的公式。其他地方与我们上一篇文章训练过程一模一样。当然需要注意的是，w, b 也不需要我们自己定义了，用 `model.variables`即可取出模型中所有参数。完整代码如下

```python
for step in range(BATCHS):
  x_train, y_train = next_batch(x_data, BATCH_SIZE)
  x_train = x_train.reshape((-1,1))
  y_train = y_train.reshape((-1,1))
  with tf.GradientTape() as tape:    
    y_pred = model(x_train)
    loss = tf.reduce_mean((y_pred - y_train)**2)
  grads = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(zip(grads, model.variables))
  if step%1000 == 0:
    print("Step: {} loss: {}".format(step,loss.numpy()))
```

*code from [my github repo](https://github.com/zht007/tensorflow-practice/blob/master/1_Linear_Regression/04_Regression_TF_2_0.ipynb) with MIT lisence*

> 需要注意的是 Keras 的输入shape 需要是二维的，所以对输入数据进行了 reshape.

### 3. MINST 的分类问题

同样的，我们也可以用这种方法训练分类问题。这里我们还是用经典的手写数字识别项目，用 CNN 神经网络结构完成训练任务。

#### 3.1 模型搭建

与线性回归的模型搭建一模一样，只不过增加了一些 CNN 层而已。这里我省略掉用于生成 batch 的帮助函数，感兴趣的读者可以移步到 Tensorflow 1.0 的文章

```python
class CNNModel(tf.keras.Model):
  def __init__(self):
    super().__init__()
    # self.Dense1 = layers.Dense(units=10, activation='relu')
    self.conv1 = layers.Conv2D(filters=12, kernel_size=(6,6), strides=(1,1), padding='same', activation='relu')
    self.conv2 = layers.Conv2D(filters=24, kernel_size=(5,5), strides=(2,2), padding='same', activation='relu')
    self.conv3 = layers.Conv2D(filters=48, kernel_size=(4,4), strides=(2,2), padding='same', activation='relu')
    self.flatten = layers.Flatten()
    self.dense = layers.Dense(units=10,activation='softmax')

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    output = self.dense(x)
    return output
```

*code from my [github repo](https://github.com/zht007/tensorflow-practice/blob/master/4_Clasification_DigitRecognizer/6_DL_Multi_Layer_CNN_for_DigitRecognizer_TF2_0.ipynb) with MIT license*

#### 3.2 训练模型

训练过程与线性回归没有什么差别，唯一需要注意的是分类问题中我们要使用 **交叉熵(cross entropy)** 来计算loss。`categorical_crossentropy`和`sparse_categorical_crossentropy`两者的区别我也在之前的这篇文章解释过。由于我们的数据已经被 one-hot 过了，所以直接使用`categorical_crossentropy`就可以了。

完整代码如下

```python
for epoch in range(num_epochs):
  X, y_true = ch.next_batch(batch_size)
  with tf.GradientTape() as tape:
    y_pred = model(X)
    loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
    loss = tf.reduce_mean(loss)

  grad = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
  
  if epoch%100 == 0:
    print("epch: {}, loss: {}".format(epoch, loss.numpy()))
```

*Code from my [github repo](https://github.com/zht007/tensorflow-practice/blob/master/4_Clasification_DigitRecognizer/6_DL_Multi_Layer_CNN_for_DigitRecognizer_TF2_0.ipynb) with MIT license*

### 3.3 模型评估

评估模型我们可以用 `tf.keras.metrics` 的评估工具进行评估。对于分类问题，我们这里实例化 `CategoricalAccuracy` (如果数据没有被 one hot encoded 需要实例化 `SparseCategoricalAccuracy`)，具体步骤如下

> 1. 实例化 SparseCategoricalAccuracy 为 比如: categorical_accuracy
> 2. 模型预测出测试数据   `y_test_pred = model.predict(x = ch.test_images)`
> 3. `SparseCategoricalAccuracy` 的 `update` 方法评估 `y_test_pred` 和真实 `y_test`的差距（正确率）

代码如下：

```python
categorical_accuracy = tf.keras.metrics.CategoricalAccuracy()

y_pred_test = model.predict(x = ch.test_images)
categorical_accuracy.update_state(y_true = ch.test_labels, y_pred=y_pred_test)
accuracy = categorical_accuracy.result().numpy()
print("The accuracy is:{}".format(accuracy))
```

当然我们也可以一边训练一边评估模型：

```python
for epoch in range(num_epochs):
  X, y_true = ch.next_batch(batch_size)
  with tf.GradientTape() as tape:
    y_pred = model(X)
    loss = tf.keras.losses.categorical_crossentropy(y_true=y_true, y_pred=y_pred)
    loss = tf.reduce_mean(loss)

  grad = tape.gradient(loss, model.variables)
  optimizer.apply_gradients(grads_and_vars=zip(grad, model.variables))
  
  y_test_pred = model.predict(x = ch.test_images)
  categorical_accuracy.update_state(y_true = ch.test_labels, y_pred=y_test_pred)
  accuracy = categorical_accuracy.result().numpy()

  if epoch%100 == 0:
    print("epch: {}, loss: {}, accuracy: {}".format(epoch, loss.numpy(),accuracy))
```

*code from my [github repo](https://github.com/zht007/tensorflow-practice/blob/master/4_Clasification_DigitRecognizer/6_DL_Multi_Layer_CNN_for_DigitRecognizer_TF2_0.ipynb) with MIT license*

最后我们在训练了 1000 个batch 之后 正确率达到了97%，与之前使用 Keras Sequancial 和 Tensorflow 1.0 的结果类似。

### 4. 总结

Tensorflow 2.0 建立模型和训练模型的过程非常简洁自然，再也不需要建立 graph 启动 Session 了，具体步骤如下：

> 1. 如果手写计算公式或神经网络结构，手动定义并初始化**参数**，注意每一层参数的 Shape.
> 2. 如果用 Keras 预先定义的 layers 搭建神经网络，需继承`tf.keras.Model`，并建立自己的**模型**。
> 3. 使用 `with tf.GradientTape() as tape:` 记录计算过程，并在过程中计算 **loss** 函数
> 4. 使用 `tape.gradient` 对**loss** 的 **参数**求**偏微分**
> 5. 定义 optimizer 并使用 `optimizer.apply_gradients` 自动更新**参数**

---

相关文章

[Tensorflow 2.0 快速入门 —— 自动求导与线性回归](https://steemit.com/cn-stem/@hongtao/55x5e5-tensorflow-2-0)

[Tensorflow 2.0 轻松实现迁移学习](https://steemit.com/cn-stem/@hongtao/tensorflow-2-0)

[Tensorflow入门——Eager模式像原生Python一样简洁优雅](https://steemit.com/cn-stem/@hongtao/tensorflow-eager-python)

------

同步到我的简书
https://www.jianshu.com/u/bd506afc6fc1