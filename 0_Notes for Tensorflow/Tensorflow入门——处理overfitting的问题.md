# Tensorflow入门——处理overfitting的问题

在之前的文章中，我们发现训练组(篮)和验证组(红)的损失函数在20个Epoch之后，向着相反方向变化，训练组损失函数继续下降，验证组损失函数反而在上升，这就是典型的Overfitting(过拟合)现象。

![](https://ws2.sinaimg.cn/large/006tKfTcgy1g18l4i0ja7j30al0703z2.jpg)

过拟合就是模型过度地学习了训练集的特征，反而没法处理测试集中更一般化的问题。处理过拟合最根本的解决方法当然是获得更多的训练样本。

但是在无法获得更多的训练样本的时候，也有两个最简单的方法，一是对权重进行正则化处理，二就对神经元随机dropout.

在Keras中我们只需要对模型进行简单改造就能实现正则化和dropout，同样的，为了方便与读者交流，所有的代码都放在了这里：

https://github.com/zht007/tensorflow-practice/tree/master/2_Classification



## L1，L2正则化

模型Overfiting其中一个原因就是某些权重在训练的过程中会被放大，L1正则化相当于给权重加了惩罚因子，从而限制了某些权重过度膨胀。L2相当于对L1惩罚因子乘了个平方，对权重的膨胀加大了惩罚力度。

在Keras的模型中引入L1，或者L2也非常简单，只需要在建立模型的时候加入:

```python
kernel_regularizer = keras.regularizers.l1
或
kernel_regularizer = keras.regularizers.l2
```

模型如下所示

```python
model = Sequential()
model.add(Dense(20,input_shape = (X_train.shape[1],),
                activation = 'relu', 
                kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(Dense(20,input_shape = (X_train.shape[1],),
                activation = 'relu', 
                kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(Dense(10,activation = 'relu',
               kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(Dense(2, activation = 'softmax'))
```

## Dropout

在需要Dropout的Dense层之后加上：

```python
model.add(Dense(2, activation = 'softmax'))
```

最后我们看看加上L2正则化和Dropout之后的模型是怎么样的。

```python
model = Sequential()
model.add(Dense(20,input_shape = (X_train.shape[1],),
                activation = 'relu', 
                kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(20,input_shape = (X_train.shape[1],),
                activation = 'relu', 
                kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(10,activation = 'relu',
               kernel_regularizer = keras.regularizers.l2(0.001)))
model.add(keras.layers.Dropout(0.5))
model.add(Dense(2, activation = 'softmax'))
model.summary()
```

Model.summary可以查看整个模型的架构和参数的个数

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_19 (Dense)             (None, 20)                180       
_________________________________________________________________
dropout_7 (Dropout)          (None, 20)                0         
_________________________________________________________________
dense_20 (Dense)             (None, 20)                420       
_________________________________________________________________
dropout_8 (Dropout)          (None, 20)                0         
_________________________________________________________________
dense_21 (Dense)             (None, 10)                210       
_________________________________________________________________
dropout_9 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_22 (Dense)             (None, 2)                 22        
=================================================================
Total params: 832
Trainable params: 832
Non-trainable params: 0
_________________________________________________________________
```



## 训练结果

最后我们看看正则化和Dropout后端的训练结果吧，是不是比之前漂亮多了。

![](https://ws2.sinaimg.cn/large/006tKfTcgy1g18lqpke82j30al0703yx.jpg)





------

同步到我的简书和Steemit
<https://www.jianshu.com/u/bd506afc6fc1>

<https://steemit.com/@hongtao>

