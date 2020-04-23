

![img](https://images.unsplash.com/photo-1575843456098-25dc4244e9f2?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

*Image from [unsplash.com](https://unsplash.com/photos/SuwFvHU-z-o) by Adrian Cuj*

前面的文章我们讨论了机器学习的各种问题，但是还没有认真讨论一下数据加载的问题。作为机器学习的第一步，如何正确和快速地加载数据以及预处理对于机器学习的项目成败是至关重要的。

我们已经很熟悉用 `pandas.read_csv` 来读取csv文件数据，然后用其内建函数或借助 **numpy** 和 **scikit-learn** 的框架来预处理数据。但这些框架并没有提供一个很方便的随机打散(shuffle) 以及批训练(batch) 的方法，以至于在处理MINST数据的时候，我们不得不自己手动写一个[预处理的帮助类](https://github.com/zht007/tensorflow-practice/blob/master/4_Clasification_DigitRecognizer/6_DL_Multi_Layer_CNN_for_DigitRecognizer_TF2_0.ipynb)，来解决这个问题。

在 Tensorflow 2.0 中，我们其实不用浪费大量时间在数据的加载和预处理上面，这篇文章我们就来介绍一下如何利自动加载经典数据集，以及用`tf.data.Dataset`对数据进行预处理。

### 1. 经典数据集简介和加载

`tf.kears`或`keras`（注意，如果未特别注明，后文中所有关于 keras 的介绍均表示 `tf.kears`）提供了若干个经典数据集分别为：

* MNIST 手写数字识别

  > 手写数字识别0-9，10个数字，train/test 分别为 60,000/10,000  张 28x28像素单通道黑白图片。

* MNIST 时装类别识别

  > 10个时装类别，train/test 分别为 60,000/10,000  张 28x28像素单通道黑白图片。

* CIFAR10/100 图片分类

  > 10/100 个图片类别， train/test 分别为 50,000/10,000  张 32x32像素3通道彩色图片。

* 波斯顿房价预测

  > 1970 年代波斯顿地区房价，包含13个特征类别。

* IMDB 影评分类

  > 一共25,000个IMDB 影评，包含正面和负面两个类别。

* 路透社新闻分类

  > 一共11,228 篇报道，包含46个话题。

加载这些数据集非常简单，一句代码搞定

```python
(x, y),(x_test, y_test) = keras.datasets.mnist.load_data()
```

训练集和验证集就分别加载到了`(x, y`)和`(x_test,y_test)`中了，上面以加载MNIST 手写数字识别数据集为例，如果需要加载其他数据集，仅需要将 `mnist`替换成需要的数据集即可，更多详细介绍参考[官方文档](https://keras.io/datasets/#imdb-movie-reviews-sentiment-classification)。

返回的数据类型为` numpy arrary` 所以可以用 numpy 的所有方法查看数据的shape, 最大最小等等。

```python
print(x.shape, y.shape, x_test.shape, y_test.shape)
print(x.min(), x.max(),x.mean())

--output---
((60000, 28, 28), (60000,), (10000, 28, 28), (10000,))
(0, 255, 33.318421449829934)
```



### 2. Dataset 转换

我们可以方便地将 numpy 数据通过`tf.data.Dataset.from_tensor_slices()`转换成 Dataset 对象。

```python
ds_train = tf.data.Dataset.from_tensor_slices((x, y))
ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
```

Dataset 对象和方便地迭代，预处理，以及多线程处理等，例如

```python
res = next(iter(ds_test))
print(res[0].shape, res[1].shape)

--output---
(TensorShape([32, 32, 3]), TensorShape([1]))
```



### 3. 数据预处理

我们可以方便地将Dataset 对象随机打散，批处理。

```python
ds_train = ds_train.shuffle(buffer_size =1000) 
# 参数 buffer_size 为缓冲大小
ds_train = ds_train.batch(100) 
# 100为每一批样本数量，相当于在原数据头部插入一个 100 的维度，整个数据的长度缩短100倍。
```

更重要的是，我们可以通过 `Dataset.map(f)` 的功能，对数据集中每个元素应用函数f, 从而同时预处理多个步骤。例如：

```python
# 定义预处理函数
def preprocess(x, y):
  x = tf.cast(x, tf.float32)/255
  y = tf.squeeze(y,axis=0)
  y = tf.cast(y, tf.int32)
  y = tf.one_hot(y, depth=10)
  return x, y

# map 预处理函数
ds_test = ds_test.map(preprocess)
```

map 之后 ds_test 就完成了数据格式转换，归一化，one_hot 等所有操作。

### 4. Tensorflow Datasets 开箱即用

其实 Tensorflow 还提供了一个更加简单的载入数据方法，而且包含了十多个常用数据集，一行代码即可下载并返回数据格式为 `tf.data.Dataset` 的对象。

以加载 MNIST 手写数字为例

```python
import tensorflow_datasets as tfds
ds2 = tfds.load(name="mnist")
```

更多关于`tensorflow_datasets`的介绍请参考[官方文档](https://www.tensorflow.org/datasets)，这里就不赘述了。

