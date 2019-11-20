![img](https://images.unsplash.com/photo-1566441367118-e92b7adb7f5b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

*image from [unsplash](https://unsplash.com/photos/yOvhvttVW5g) by Zhang Kenny*

前面的几篇文章从[线性回归](https://www.jianshu.com/p/c44705808f7e)，到[手写数字识](https://www.jianshu.com/p/e68172ba8c91)别再到[预测牛奶产量](https://www.jianshu.com/p/e2ff67c7b7aa)，我们用 Tensorflow 2.0 与 Keras 结合完成了全链接神经网络，卷积神经网络以及循环神经网络的搭建和训练。那 Tensorflow 2.0 和 Keras 到底是什么关系，我们应该如何选择和搭配二者来完成自己的项目呢？这篇文章就来探讨一下这个问题。

### 1. Tensorflow 和 Keras 的关系

从历史上来看 Keras 和 Tensorflow 是相互独立的框架。Keras 中我们编写代码的是**前端**，前端是一个高度模块化的的框架，使用者可以非常快速和方便的搭建模型和训练模型。但是，光有前端是不够的，需要**后端**框架实现计算，而 Tensorflow 就是 Keras 支持的一个后端框架。就像网页一样，前端我们能看到的是图片，文字，动画，后端就是支撑这些内容，布局内容和逻辑的代码。

虽然 Tensorflow 并不是 Keras 的唯一后端框架，但是却是最流行的框架。后来融入tensorflow 中的`tf.keras` 则是只能用 Tensorflow 后端的Keras 框架。在 Tensorflow 的高级API 中 Keras 也逐渐取代 `tensorflow.layers` 以及 `tensorflow.estimator`成为Tensorflow 2.0 中几乎**唯一**的高级API。

### 2. 不使用 Keras

在 Tensorflow 2.0 中，是否可以在不使用 Keras 的情况下完成模型的搭建和训练？答案当然是肯定的，在线性回归的例子中，我们就没有用到 Keras 而是使用 Tensorflow 定义和训练模型的。这个时候和 Tensorflow 1.0 是差不多的，必须定义清楚每一层神经网络的计算公式，参数 shape，层与层之间的计算与连接等等。

在训练的时候，需要注意采用 Tensorflow 2.0 标准的模式既：

>1. 在 `with tf.GradientTape() as tape:` 的包裹下，记录模型的预测结果，损失函数，用于稍后求导。
>2. `tape.gradient(loss, 参数)` 计算 loss 对 参数的导数。
>3. 定义的 `optimizer` 后向求导对参数进行优化。

### 3. Keras 自定义模型和层

我们知道，无论是全链接的神经网络还是 CNN， RNN，都有标准的结构和连接方式。我们没有必要重新将轮子发明一遍，于是我们可以通过继承 `keras.Model` 的方式自定义自己的模型。 这也是前面的文章用 CNN 分类手写数字，以及用 RNN 预测牛奶产量的例子。**自定义模型**的方法已经在[前文](https://www.jianshu.com/p/e68172ba8c91)介绍和总结过了，在这里就不赘述了。

同样的，我们还可以通过继承 `tf.layers.Layer` 来自定义我们需要的层。

>自定义层分为两步
>
>1. 继承 `keras.layers.Layer` 类，在初始化函数 `__init__` 中设置特征长度 inp_dim 和输出特征长度 outp_dim.
>2. 并通过self.add_variable(name, shape)创建 shape 大小,名字为 name 的张量,并设置为需要优化.
>3. 定义 前向运算逻辑 def call 方法

如下面的例子我们定义一个没有偏置的全链接层：

```python
class MyDense(tf.keras.layers.Layer):     # 自定义网络层     
  def __init__(self, inp_dim, outp_dim):         
    super(MyDense, self).__init__()         # 创建权值张量并添加到类管理列表中,设置为需要优化         
    self.kernel = self.add_variable('w', [inp_dim, outp_dim], trainable=True)
   
  # 实现自定义类的前向计算逻辑 
    def call(self, inputs, training=None):                         
      out = inputs @ self.kernel    # X@W       
      out = tf.nn.relu(out)         # 执行激活函数运算
      return out
```

之后我们就可以像使用 keras 自带的layer 一样使用我们自定义的 layer：`MyDense`。这个Layer 既可以封装在 Keras 的 `Sequential`容器中，也可以放在**自定义的模型**中。

### 4. Sequential 容器

在 Keras 中最简洁的模型搭建和训练的方法就是将 layers 封装在 `Sequential`的容器中，然后调用模型装配(compile)与训练(fit)的高层接口来实现。 

这种方法其实基本上与 Tensorflow 本身没有太大关系了，只需要对 Keras 前端进行操作。前面很多文章已经介绍过具体的操作方法，这里就简单总结一下：

> 1. 封装模型mdel在 `squential` 中.
> 2. model.compile()  装配模型，在这里选择**优化器**，定义**损失函数**以及测量指标
> 3. modle.fit() 训练模型，在这里可以选择训练的输入和标签，batch size, 循环(epoch)次数，以及验证数据组等。
> 4. 模型训练之后可以用 model.predict() 和 model.evaluate() 来预测和评估模型

### 总结

可以看出 Tensorflow 2.0 已经与 Keras 深度进行了融合，我们既可以用 Keras 的高级接口迅速搭建和训练模型，也可以完全抛弃 Keras 使用 Tensorflow 一步一步定义搭建和训练模型。当然最好是结合两者的优缺点，比如自定义层和模型来灵活和快速的搭建和训练自己的模型。