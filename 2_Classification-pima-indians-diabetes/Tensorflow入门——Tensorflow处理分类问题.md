# Tensorflow入门——Tensorflow处理分类问题，Classification with Tensorflow



[上一篇文章](https://steemit.com/cn-stem/@hongtao/tensorflow-keras-classification-with-keras)我们介绍了如何使用Keras处理分类问题，那Tensorflow可不可以像[处理回归问](https://steemit.com/cn-stem/@hongtao/tensorflow)题一样，直接处理分类问题呢？

答案当然是肯定的。这篇文章我们就用[之前](https://steemit.com/cn-stem/@hongtao/tensorflow-keras-classification-with-keras) 相同的数据，来学习如何用Tensorflow训练一个线性分类器。数据预处理的过程就略过了，可以参考[上一篇文章](https://steemit.com/cn-stem/@hongtao/tensorflow-keras-classification-with-keras)。

同样的，为了方便与读者交流，所有的源代码都放在了这里：https://github.com/zht007/tensorflow-practice/tree/master/2_Classification>



## 1. 定义数据shape

在Keras中，我们只需要考虑数据的输入和输出shape，中间的Shape以及参数的Shape，Keras都可以自动帮我们搞定。然而在Tensorflow就必须手动定义参数的Shape了。

首先我们还是要借助keras的工具将标签y转换成one hot 的数据。

```python
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
```

在定义权重W的Shape的时候，不妨教大家一个技巧，

Rows:       W.shape[0] = X.shape[1],  (输入的feature数)；

columns: W.shape[1] = Y.shape[1],  (输出的classfication数)

```python
n_features = X_train.shape[1]
n_classes = y_train_cat.shape[1]

w_shape = (n_features, n_classes)
b_shape = (1, n_classes)
```

## 2. Variables和Placeholders 

参数W和b是Variables，要训练的X和Y是Placeholders

```python
W = tf.Variable(initial_value = tf.random.normal(shape = w_shape))
b = tf.Variable(initial_value = tf.random.normal(shape = b_shape))

X = tf.placeholder(tf.float32)
y_true = tf.placeholder(tf.float32)
```

## 3. 计算图谱Graph

与线性回归一样，线性分类器的计算图谱如下

```python
y_hat = tf.matmul(X,W) + b
```



## 4. 损失函数和Optimizer

损失函数需要选择softmax_cross_entropy，Optimizer与线性回归一样，用梯度下降Optimizer就OK了。

```
loss = tf.losses.softmax_cross_entropy(y_true, y_hat)

optimizer = tf.train.GradientDescentOptimizer(0.05)
train = optimizer.minimize(loss)
```

## 5. Session中训练

初始化Vaiable之后就可以在Session中进行训练啦。为了实现在Keras中存储损失函数的记录的功能，我们手动定义了字典history，用来存储训练组和验证组的损失函数变化过程。

```python
epochs = 50000
history = {'loss':list(),'val_loss':list()}
with tf.Session() as sess:
    sess.run(init)
    
    for epoch in range(epochs):
        sess.run(train,{X:X_train, y_true:y_train_cat})
        
        history['loss'].append(sess.run(loss, {X: X_train, y_true: y_train_cat})) 
        history['val_loss'].append(sess.run(loss, {X: X_test, y_true: y_test_cat})) 
        
        if epoch % 100 == 0:
            print("Iteration {}:\tloss={:.6f}:\tval_loss={:.6f}"
                  .format(epoch, history['loss'][epoch], history['val_loss'][epoch]))
            
    y_pred = sess.run(y_hat, {X: X_test})
    W_final, b_final = sess.run([W, b])
```

## 6. 验证结果

最后我们将训练结果可视化，可以看到效果还不错，损失函数的下降曲线非常平滑，而且训练集和测试集的损失函数也相差不大。

![](https://ws1.sinaimg.cn/large/006tKfTcgy1g1778h15byj30bu09o3yv.jpg)

------

同步到我的简书和Steemit
<https://www.jianshu.com/u/bd506afc6fc1>

<https://steemit.com/@hongtao>