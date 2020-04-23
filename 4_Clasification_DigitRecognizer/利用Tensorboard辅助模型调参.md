![img](http://ww4.sinaimg.cn/large/006tNc79gy1g45resx595j30rs0jvjwx.jpg)

Image source from unsplash by [Timothy L Brock](https://unsplash.com/@timothylbrock)

上一篇文章介绍了如何在 [Keras 中调用 Tensorboard](https://steemit.com/cn-stem/@hongtao/keras-tensorboard)。这篇文章就来谈谈如何用 Tensorboard 帮助模型调参。

代码repo见这里

https://github.com/zht007/tensorflow-practice

还是用[手写数字MINST数据集](https://steemit.com/cn-stem/@hongtao/2jkjay-tensorflow-mnist)为例，之前我们通过CNN的模型将识别率提高到了99%，CNN网络中的各个参数是怎么得到的呢，多少层卷积层，多少层全连接层，每层神经网络多少个神经元或者多少个Filter呢？如何调整这些参数以保证模型是具有**"识别"**手写数字的能力，而并不是仅仅将每个图片对应的数字简单粗暴地**"记"**下来了呢？

这里我们就需要**遍历**不同参数的组合，然后使用 Tensorboard 可视化的工具找出最佳的参数组合。

### 1. 提取模型参数

最容易调节的参数：卷积层层数，每层神经元个数(Filter 数量) 和 全连接层层数，这几个参数分别list三个数。

```python
dense_layers = [0,1,2]
layer_sizes = [32, 64,128]
conv_layers = [1, 2, 3]
```

### 2. 建立和训练各个模型

三个参数，三个for循环遍历，一共建立并训练9个模型。注意: tensorboard 需要在循环中调用。

```python
NAME = "{}-conv-{}-notes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
tensorboard = TensorBoard(log_dir='gdrive/My Drive/dataML/logs1/{}'.format(NAME))
```

当然为了提高速度，我们只训练了 30 个epoch.

完整代码如下：

```python
for dense_layer in dense_layers:
  for layer_size in layer_sizes:
    for conv_layer in conv_layers:
      
      NAME = "{}-conv-{}-notes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,int(time.time()))
      tensorboard = TensorBoard(log_dir='gdrive/My Drive/dataML/logs1/{}'.format(NAME))
      print(NAME)
      
      model = models.Sequential()
      model.add(layers.Conv2D(filters = layer_size, kernel_size=(6,6), strides=(1,1),
                              padding = 'same', activation = 'relu',
                              input_shape = (28,28,1)))
      
      for l in range(conv_layer - 1):
        model.add(layers.Conv2D(filters = layer_size,kernel_size=(5,5),strides=(2,2),
                                padding = 'same', activation = 'relu'))
       
      model.add(layers.Flatten())
      
      for l in range(dense_layer):
        model.add(layers.Dense(units = layer_size, activation='relu'))
      
      
      model.add(layers.Dense(units=10, activation='softmax'))
      model.summary()
      
      adam = keras.optimizers.Adam(lr = 0.0001)

      model.compile(loss=keras.losses.categorical_crossentropy, 
                    optimizer=adam, 
                    metrics=['accuracy'])
      
      
      H = model.fit(x_train, y_train,
          batch_size=50,
          epochs=30,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard])
```

### 3. 在Tensorboard 中 查看结果

当然我们最关心的是测试集的准确率和损失函数

![image-20190618164327203](http://ww1.sinaimg.cn/large/006tNc79gy1g45qnjl1tfj30ow124tii.jpg)

一共9个结果，看起来比较麻烦，可以通过左下角的工具，可以勾选自己想看的结果。通过对比，可以发现卷积层操过三层，神经元或 Filter 数量操过64个，全连接层超过2个，就会出现明显的过拟合现象。

### 4. 调整参数组合优化模型

通过 Tensorboard 的观察，我们继续优化模型参数，这次可以去掉过造成拟合的参数，增加batch size

```python
dense_layers = [1,2]
layer_sizes = [32,64]
conv_layers = [2]
batch_sizes = [50,100,200]
```

重复上述过程，进一步优化参数，去掉造成过拟合的参数，增加Learning Rate 

```python
dense_layers = [1,2]
layer_sizes = [32,64]
conv_layers = [2]
batch_sizes = [50,100]
learning_rates = [0.0005,0.0001,0.00005]
```

进一步缩小遍历的参数范围，增加训练的 epoch 数量，最终得到一组自己满意的参数组合

```python
dense_layers = [1]
layer_sizes = [32]
conv_layers = [2]
batch_sizes = [100]
learning_rates = [0.0005]
```

### 5. 总结

机器学习模型调参的过程实际上是一个不断尝试的过程，将想要调整的参数列出来一一训练。然后借助 Tensorboard 缩小探索的范围，最终得到一个自己满意的参数组合。

-----

参考资料

[1]https://www.kaggle.com/c/digit-recognizer/data

[2]https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0

[3]https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git

[4]https://www.tensorflow.org/api_docs/

------

相关文章

[Tensorflow入门——单层神经网络识别MNIST手写数字](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)

[Tensorflow入门——多层神经网络MNIST手写数字识别](https://steemit.com/cn-stem/@hongtao/6qe2nw-tensorflow-mnist)

[AI学习笔记——Tensorflow中的Optimizer](https://steemit.com/tensorflow/@hongtao/ai-tensorflow-optimizer)

[Tensorflow入门——分类问题cross_entropy的选择](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)

[AI学习笔记——Tensorflow入门](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)

[Tensorflow入门——Keras简介和上手](https://steemit.com/cn-stem/@hongtao/tensorflow-keras)

---

同步到我的简书
https://www.jianshu.com/u/bd506afc6fc1