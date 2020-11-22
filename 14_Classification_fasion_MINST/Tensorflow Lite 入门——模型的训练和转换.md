---
title: Tensorflow Lite 入门——模型的训练和转换
date: 2020-11-22 17:29:00
tags: [ML, TF2.0, TFLite]
---



之前的文章中介绍的使用 Tensorflow 训练的模型要么只能运行在 PC 端，要么需要云端的支持。随着智能手机和物联网设备的普及，能够在智能手机甚至嵌入式设备直接运行的模型需求就越来越高。这篇文章就开始介绍 Tensorflow Lite, 这个能够运行在智能手机和 嵌入式设备的开源深度学习框架。

通常，我们会在 PC 或者云端建立模型，并对模型进行训练，然后将模型转换成 Tensorflow Lite 的格式，并最终部署到终端设备上，这篇文章我们就用 Fashion MNIST 的数据集，建立并训练模型，并采用模拟器的方式部署到终端设备上进行测试。

### 1. 数据的加载和训练

这个部分的内容与之前Tensorflow 2.0 快速入门内容重复，在这里就不过多赘述了。但是值得注意的是，之前的文章我们都是使用的  keras 的数据集，其数据格式是 numpy。Fasion MINST 是从 `tensorflow_datasets` 中直接加载的。将数据集分为 80% 训练集，10% 测试集和10%验证集。tfds的详细使用说明请参考[官方文档](https://www.tensorflow.org/datasets/api_docs/python/tfds/load)。

```python
import tensorflow_datasets as tfds

splits, info = tfds.load('fashion_mnist', with_info=True, as_supervised=True, 
                         split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'])

(train_examples, validation_examples, test_examples) = splits

num_examples = info.splits['train'].num_examples
num_classes = info.features['label'].num_classes
```

数据经过预处理之后，我们使用 Keras 的 API 快速搭建了一个五层的 CNN 神经网络并使用 `model.fit` 对模型进行了训练。

### 2. 模型的保存与转换

训练好的模型这里使用了 tf.save_model.save() 将模型保存在了指定目录。

```python
export_dir = 'saved_model/1'

# Use the tf.saved_model API to export the SavedModel
tf.saved_model.save(model, export_dir)
```

本文的重点是模型转换，在Tensorflow Lite 中，使用 `TFLiteCoverter` 可以轻松将模型转换成 Tensorflow Lite 的模型。

```python
# Use the TFLiteConverter SavedModel API to initialize the converter
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir=export_dir)

# Invoke the converter to finally generate the TFLite model
tflite_model = converter.convert()

# Save the model file as 'model.tflite'
tflite_model_file = 'model.tflite'
with open(tflite_model_file, "wb") as f:
  f.write(tflite_model)
```

### 3. 模型的优化

converter 在默认的情况下是将模型权重从32位浮点数转换成8位整数从而大大减小模型的大小。

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
```

我们也可以将模型手动调整为16位浮点数

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

当然除了 default 模式，官方也提供了精度优先和大小优先的优化模式，详细内容参考[官方文档](https://www.tensorflow.org/lite/performance/post_training_quantization)

```python
tf.lite.Optimize.OPTIMIZE_FOR_LATENCY
tf.lite.Optimize.OPTIMIZE_FOR_SIZE
```

### 4. 模型测试

到这里，实际上我们就可以将转换成 Tensorflow Lite 的模型部署到设备上进行测试了，但是此时，我们并不知道模型的性能如何，Tensorflow Lite 提供了模拟器，我们可以轻松部署在模拟器上对转换后的模型进行测试。

测试模型分为三步，

> 第一步：加载 TFLite 模型，部署tensor
>
> 第二步：获取 input 和 output index	
>
> 第三步：加载数据并获取结果

```python
# Step 1: Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Step 2: Get input and output tensors index
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

# Step 3: Get results
interpreter.set_tensor(input_index, img)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)
```

### 5. 总结

Tensorflow Lite 在PC 或者云端的训练和测试可以分为三个步骤： 1. 数据加载，模型搭建与的训练。2. 模型的保存与转换。3.模型的测试。

经过这三个步骤之后，我们就可以将转换后的模型部署在终端设备上了，Tensorflow Lite 不仅可以支持 Android 和 iOS 的智能手机也支持raspberry pi 智能手环这样的嵌入式设备，如果有机会将在后面的文章中给大家介绍。

---

相关文章

[强化学习——轻松用 Keras 一步一步搭建DQN模型](https://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483840&idx=1&sn=c17099fc9edce78e8055e086f84c8be6&chksm=fd828e53caf50745d03126b4e85361a3a8cfa4defb2284584e6317eae78d007f995cd5136500&token=698937485&lang=zh_CN&scene=21#wechat_redirect)
[DQN——图解并轻松上手深度Q-Learning](https://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483831&idx=1&sn=61073043c3d2ad9f6a5e85347a10fac1&chksm=fd828e24caf5073290c9ac5dae2bd005097a0d4612ca176a57ea093045dca4993799354703e9&token=698937485&lang=zh_CN&scene=21#wechat_redirect)
[AI学习笔记——MDP(Markov Decision Processes马可夫决策过程)简介](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483821&idx=2&sn=4726fa197e78755891b2b681412e75bf&chksm=fd828e3ecaf50728d2597339ced0ec6a0664f51cbab6472c5a25160b532ce87dd871b1b9ba38&token=466152961&lang=zh_CN&scene=21#wechat_redirect)
[AI学习笔记——Q Learning](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483786&idx=1&sn=e556e3e4e496b7327243d21888633c7a&chksm=fd828e19caf5070fa54e91ce7c66622d11f3e3e25d53519945465aec6b0debad3b4ea1a232c5&token=466152961&lang=zh_CN&scene=21#wechat_redirect)
[AI学习笔记——Sarsa算法](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483811&idx=2&sn=9971b5573f8fba3fb1b28a8725c33f24&chksm=fd828e30caf50726640e84039233759525662fbe73c24db7c15577dec9888263fbf56ed71371&token=466152961&lang=zh_CN&scene=21#wechat_redirect)
[AI学习笔记——卷积神经网络（CNN）](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483720&idx=4&sn=5f2d6075cd43e0efad614d2f0199a44b&chksm=fd828edbcaf507cde0f73fce3ca2bc3fef8b525764c51fd35e9a2f714109f5e6a4f25531371c&token=466152961&lang=zh_CN&scene=21#wechat_redirect)