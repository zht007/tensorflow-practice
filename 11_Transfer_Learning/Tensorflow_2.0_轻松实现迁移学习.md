![pink and white rose lot](https://images.unsplash.com/photo-1456415333674-42b11b9f5b7b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

*image from unsplash by [Gábor Juhász](https://unsplash.com/@juhg)*

迁移学习即利用已有的知识来学习新的知识，与人类类似，比如你学会了用笔画画，也就可以学习用笔来画画，并不用从头学习握笔的姿势。对于机器学习来说，可以用现有的已经经过训练的模型，来训练我们自己的模型，并没有必要从头训练。

以图像识别的神经网络模型为例，我们可以使用已经在 Image Net 上训练过的模型直接训练我们自己的模型。这篇文章我们的目的是对5种花进行分类，Image Net 对 1000 多类物体进行过分类，所以完全能够"覆盖"我们要训练的图片类别，适合迁移学习。

### 1. Imports

因为 Tensorflow 2.0 还未正式发布，所以为了兼容性的原因需要从“_ future_” 中 import 不少库。tf.enable_eager_execution() 是开启 eager模式，在之前的文章中已经介绍过了，这里就不赘述了。tensorflow_hub 将在后面介绍。当然我们这里主要使用的还是 Keras。

```python
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

tf.enable_eager_execution()

import tensorflow_hub as hub
import tensorflow_datasets as tfds

from tensorflow.keras import layers
```

*code from [github](https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb) with  Apache License*

### 2. Tensorflow Datasets

我们直接从 [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/overview#tf_flowers) 上下载花的图片数据库，数据库名字为“tf_flowers”, 数据库中只有一个 TRAIN 数据集，需要分类。这里我们按照训练和验证数据集70：30 的比例分配数据。

```python
splits = tfds.Split.TRAIN.subsplit([70, 30])

(training_set, validation_set), dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True, split=splits)
```

由于图片的尺寸各不相同，我们需要统一尺寸，并且为数据制作 Batch。Tensorflow Datasets 提供了非常方便的工具。

```python
IMAGE_RES = 224

def format_image(image, label):
  image = tf.image.resize(image, (IMAGE_RES, IMAGE_RES))/225
  
  return image, label

BATCH_SIZE = 32

train_batches = training_set.shuffle(num_training_examples//4).map(format_image).batch(BATCH_SIZE).prefetch(1)

validation_batches = validation_set.map(format_image).batch(BATCH_SIZE).prefetch(1)
```

*code from [github](https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb) with  Apache License*

处理图片 .map() 实现，map中嵌套的函数 format_iamge 即resize 图片尺寸的方法。



### 3. TensorFlow Hub

[TensorFlow Hub](http://tensorflow.org/hub) 就提供了这些已经经过训练的模型，供我们免费使用。我们可以在 https://tfhub.dev/ 寻找我们需要的模型。

需要注意的是我们需要的不是完整的模型，而是模型在分类之前(没有最后一层的神经网络)的结构和参数。这种模型是 **Feature Vector**

![image-20190819100259037](http://ww4.sinaimg.cn/large/006tNc79gy1g65pac4jwuj31620tw44p.jpg)



这里我们使用的是[Mobile Net](https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4)。

* 找到 URL
* 使用 hub.KerasLayer() 导入模型

```python
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor = hub.KerasLayer(URL,
                                   input_shape=(IMAGE_RES, IMAGE_RES,3))
```

### 4. 整合 Keras 模型

在创建 Kearas 模型之前需要 Freeze 模型参数, 因为前面的参数已经训练得足够好了，我们暂时不希望在向后传播的过程中改变它们，而是重点训练最后新加的那一层。

```python
feature_extractor.trainable = False
```

建立 keras 模型的过程与之前的步骤并无差别，只是在末尾添加我们最终需要输出的那一层即可。

```python
model = tf.keras.Sequential([
    feature_extractor,
    layers.Dense(num_classes,activation="softmax")
])

model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
```

*code from [github](https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb) with  Apache License*

## 5. 训练模型

这里对 keras 模型的训练与我们熟悉的方法并无太大差异，注意的是我们这里使用的是前面tfds 处理过的数据，已经经历了 resize, shuffle 和 batch 化了。并不需要再 model.fit 中shuffle，定义 bacth_size 了。

我们仅仅用了 6 个回合就将验证训练集的正确率提高到88%。

```python
EPOCHS = 6

history = model.fit(
     train_batches,
     epochs = EPOCHS,
     validation_data = validation_batches
  )
```

*code from [github](https://github.com/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l06c03_exercise_flowers_with_transfer_learning_solution.ipynb) with  Apache License*

### 6. 总结

本文通过 Tensorflow 2.0 中提供的前瞻性模型 Mobile Net 简单地介绍了迁移学习，以及如何用Tensorflow 实现迁移学习。过程中，我们 Freeze 了已训练的模型的参数，用很少的训练时间取得了不不错的成绩。当然为了进一步提高模型的精度，我们需要解锁所有模型的参数，并对模型进一步进行精调，这部分的内容将会在放在后续的文章中。

----

相关文章

[Tensorflow入门——单层神经网络识别MNIST手写数字](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)

[Tensorflow入门——多层神经网络MNIST手写数字识别](https://steemit.com/cn-stem/@hongtao/6qe2nw-tensorflow-mnist)

[AI学习笔记——Tensorflow中的Optimizer](https://steemit.com/tensorflow/@hongtao/ai-tensorflow-optimizer)

[Tensorflow入门——分类问题cross_entropy的选择](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)

[AI学习笔记——Tensorflow入门](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)

[Tensorflow入门——Keras简介和上手](https://steemit.com/cn-stem/@hongtao/tensorflow-keras)

------

同步到我的简书
https://www.jianshu.com/u/bd506afc6fc1