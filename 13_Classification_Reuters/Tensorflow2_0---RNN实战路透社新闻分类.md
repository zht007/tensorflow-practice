![img](https://images.unsplash.com/photo-1579446343613-c533ca94afde?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

*Image from [unsplash.com by @Jason_xj]()*

之前的文章我们介绍了 [RNN 循环网络](https://www.jianshu.com/p/540946052325)，[并用循环网络](https://www.jianshu.com/p/e2ff67c7b7aa)成功地预测了牛奶产量。这篇文章我们继续使用 RNN 以及 LSTM 和 GRU 处理分类问题。我们使用的是 Keras 自带的数据集——路透社新闻分类问题。

**关注微信公众号获取源代码（二维码见文末）**

### 1. RNN 回顾

与卷积神经网络处理空间局部相关性数据不同，循环网络主要用于处理时间**序列 (Sequence)**相关的问题，既数据具有时间前后相关性，比如股市行情，语音文本等。

我们当然可以使用全连接的神经网络处时间理序列问题，但是对于该类问题循环网络相对于全连接网络的优势有两个：

1.RNN 可以通过共享权值，大大减少了参数数量。

> 举一个语句情感分类的例子：一段影评到底是正面还是负面。如果使用全连接的神经网络，那么我们可能会将一个评论里面的每一个单词都建一个神经网络，每个神经网络虽然可以结构相同但是参数是不同的。
>
> 循环神经网络可以共享一套权值，单词通过这一个神经网络按时间顺序输入即可。

2.全连接神经网络无法像 RNN 这样感知前文 (甚至后文) 的语义信息，导致整个句子语义丢失。

### 2. 数据集简介及导入

该数据集包含 11,228  条新闻，被标记成了 46 个类别，应该就是时政，娱乐什么之类的。模型的训练目标既为读新闻内容识别新闻类别。

数据导入跟之前没有太大差别，但是需要注意的是我们拿到的训练集 x_trian 是一个单词已经被数字编码了。

```python
(x_train, y_train),(x_test, y_test) = keras.datasets.reuters.load_data(num_words = total_words)
```

这里需要注意的是原始数据每篇新闻的长度不同，新闻是以 list 的形式存在一维 numpy array 中的。这我们需要统一新闻长度，将不足长度的地方 pad 为0。

```python
## pad sequence to the same length
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen = max_news_words)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen = max_news_words)
```

数据预处理部分并没有 reshape 输入，仅仅是将标签 y 进行了 one hot 编码。

同时需要注意在 batch 处理数据的时候让 drop_remainder = True，这样可以丢弃掉最后一个数量不足 batch size 的 batch.

```python
def preprocess(x,y):
  y = tf.one_hot(y, depth=num_classes)
  return x, y

# shuffle and batch dataset and drop the last batch shorter than batch_size
ds_train = ds_train.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder = True)
ds_test = ds_test.shuffle(1000).map(preprocess).batch(batch_size, drop_remainder = True)
```

### 3. RNN模型建立

在 keras 当中我们有两种方式建立 RNN 模型，比较推荐的方式是调用 `layers.SimpleRNN` 类。该类之前文章介绍的方式一致，比较简单，不需要手动处理层与层之间的状态信息。

另一种方式是调用 `layers.SimpleRNNCell`，这种方式比较底层，需要手动处理层与层之间的状态信息。这种方式虽然麻烦，但是有利于加深我们对 RNN 的理解，所以本文以这种方式为主介绍 RNN 模型的建立。

首先，定义状态参数，没层RNN 只有一个状态参数，初始化为0，注意其 shape

```python
class RNN(keras.Model):
  def __init__(self, num_units, num_classes):
    super().__init__()
  
  	#state = [[h]] -> [[b, unites]]
    self.state0 = [tf.zeros([batch_size, num_units])]
    self.state1 = [tf.zeros([batch_size, num_units])]
```

其次，这里需要添加一个 embbeding 层，需要强调的是对于文本处理 embbeding 是必要的。其作用是将输入的单词进行编码。这里我们直接使用 Keras 的 `layers.Embedding`层进行编码。

> embbeding 转化后的 shape 是这样的 [batch_size, seq_len, feature_len]，其中 seq_len 表示一个句子中有多少个单词，这里我们定义为 max_new_words。feature_len 表示编码后，一个单词的特征纬度，这里我们设置为 embedding_len。

注意这一层也是参与训练的。

```python
    # embedding [b, 200] -> [b, 200, 100]
    self.embedding = layers.Embedding(total_words, embedding_len, input_length=max_news_words)

```

然后，定义两层带 dropout 的 RNN 层，和最后一个全连接的输出层

```python
    self.RNNcell0 = layers.SimpleRNNCell(num_units, dropout=0.5)
    self.RNNcell1 = layers.SimpleRNNCell(num_units, dropout=0.5)
    self.fc = layers.Dense(num_classes)
```

最后，当然是定义 call 函数。这里注意两点

1. 由于带有 dropout 所以需要传入 training 参数，以便区分训练和验证两种状态。
2. State 参数需要手动更新。

```python
def call(self, inputs, training = None):


    #[b, max_news_words] -> [b, max_news_words, embedding_len]
    words = self.embedding(inputs)

    # for every words in sequence
    # [b, max_news_words, embedding_len] -> unstack [b, embedding_len]

    state0 = self.state0
    state1 = self.state1

    for word in tf.unstack(words, axis = 1):
      # state1 = out1 = x*w_x + state0*w_state + b
      # state out [b, num_units]
      out0, state0 = self.RNNcell0(word, state0, training)
      out1, state1 = self.RNNcell1(out0, state1, training)

    outputs = self.fc(out1)

    return outputs
```

### 4. 模型的训练可视化

模型的训练与之前相同既可以使用 Keras 封装好的，compile 和 fit 方法，也可以使用更加灵活的 Tensorflow 2.0 就不在这里赘述了，`tf.GradientTape()` 方式。不过注意的是在使用  model.fit 方式的时候需要设置一个参数 experimental_run_tf_function=False， 否则会报错。我也不知道什么原因，官方目前也没解释。

可视化的化，当然推荐使用 Tensorboard，祥见[前文](https://www.jianshu.com/p/c9277c3f4371)。

### 5. 使用 LSTM 和 GRU

将 Simple RNN 替换成 LSTM 和 GRU 的方法非常简单，基本上就是将 `layers.SimpleRNNCell` 替换成 `layers.LSTMCell`和`layers.GRUCell`即可。但是需要注意的是 LSTM 状态信息参数是两个 h 和 c。 所以初始化 state 参数的时候两个参数都需要初始化。

```python
# state=[h,c] =[[b, units],[b,units]]
    self.state0 = [tf.zeros([batch_size, num_units]), tf.zeros([batch_size, num_units])]
    self.state1 = [tf.zeros([batch_size, num_units]), tf.zeros([batch_size, num_units])]
```

----

相关文章

[Tensorflow 2.0 --- ResNet 实战 CIFAR100 数据集](https://www.jianshu.com/p/30173def8a99)

[Tensorflow2.0——可视化工具tensorboard](https://www.jianshu.com/p/c9277c3f4371)

[Tensorflow2.0-数据加载和预处理](https://www.jianshu.com/p/b796823ad32c)

[Tensorflow 2.0 快速入门 —— 引入Keras 自定义模型](https://www.jianshu.com/p/e68172ba8c91)

[Tensorflow 2.0 快速入门 —— 自动求导与线性回归](https://www.jianshu.com/p/c44705808f7e)

[Tensorflow 2.0 轻松实现迁移学习](https://www.jianshu.com/p/54aa43935c2b)

[Tensorflow入门——Eager模式像原生Python一样简洁优雅](https://www.jianshu.com/p/883addf4a1b2)

[Tensorflow 2.0 —— 与 Keras 的深度融合](https://www.jianshu.com/p/9a5ae5d92dba)



欢迎扫描二维码关注我的微信公众号“tensorflow机器学习”，一起学习，共同进步

![image-20200120124818801](https://tva1.sinaimg.cn/large/006tNbRwgy1gb3bj11ljqj30p00p2drs.jpg)