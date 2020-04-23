![](https://steemitimages.com/0x0/https://images.unsplash.com/photo-1542378993-3aa1366b0090?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

_Image source: [unsplash.com](https://images.unsplash.com/photo-1542378993-3aa1366b0090?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1950&q=80) by Sergey Pesterev_

[上一篇文章](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)中，我们用Tensorflow搭建了单层神经网络，该网络对MNIST手写识别率能到达90%。如何进一步提高识别率呢？Let's go deeper, 搭建更多层的神经网络吧。

同样的，为了方便与读者交流，所有的代码都放在了这里：

#### Repository:

[https://github.com/zht007/tensorflow-practice](https://github.com/zht007/tensorflow-practice)

## 1. 初始化W和B

权重(Weight)W和偏置(Bias)B的shape，是由每一层神经元的个数决定的，输出层的神经元个数保证与Class的数量一致(10个)，输入层和隐藏层的神经元数目是没有固定的要求的。

多层神经网络实际上就像是在单层神经网络的基础上"叠蛋糕"。这里我们设计5层的神经网络，神经元个数从输入到输出分别为200，100，60，30和10个。

```python
L = 200
M = 100
N = 60
O = 30
```

同样的，首先初始化每一层的权重和偏置。注意这里使用tf.truncated_normal方法随机初始化W。偏置B要尽量选择一个较小的非零数值来初始化，以适应激活函数RELU最有效区间。

```python
W1 = tf.Variable(tf.truncated_normal([784, L], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([L])/10)

W2 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B2 = tf.Variable(tf.ones([M])/10)

W3 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B3 = tf.Variable(tf.ones([N])/10)

W4 = tf.Variable(tf.truncated_normal([N, O], stddev=0.1))
B4 = tf.Variable(tf.ones([O])/10)

W5 = tf.Variable(tf.truncated_normal([O, 10], stddev=0.1))

B5 = tf.Variable(tf.zeros([10]))
```

*该部分代码部分参考[[2]](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)[[3]](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git) with Apache License 2.0*

## 2. 搭建神经网络

搭建神经网络类似于"叠蛋糕"，copy&paste输出层就好了，与输出层不同的是，在输入层和隐藏层中，我们用了比较流行的RELU激活函数。当然，输入层不要忘了Reshape。

```python
XX = tf.reshape(X,[-1,784])

Y1 = tf.nn.relu(tf.matmul(XX, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Ylogits = tf.matmul(Y4, W5) + B5
Y = tf.nn.softmax(Ylogits)
```
*该部分代码部分参考[[2]](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)[[3]](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git) with Apache License 2.0*

Optimizer的选择已经神经网络的训练与单层神经网络没有任何区别，这里就不讨论了，感兴趣的朋友可以去查看源码，接下来我们来看看这个5层神经网络的表现吧。

## 3. 识别效果

我们用Adam的优化器，0.005的学习速率，100的batch_size，训练了20000个Iteration。最后我们发现训练组的准确率几乎能达到100%，但是验证组的的准确率却始终在97%附近徘徊

```python
Iteration 19900: loss_train=0.000003: loss_val=0.128829: acc_train=1.000000: acc_val=0.978571

```

![](https://steemitimages.com/0x0/https://ws4.sinaimg.cn/large/006tKfTcgy1g1l8jdk7utj30am097q3d.jpg)

是的，这就是深度学习典型的overfitting问题。

## 4. 可变学习速率

学习速率决定了梯度下降过程中所迈的步子的大小。可以想象，如果迈的步子太大很有可能一步就跨过了最优点，最后只能在最优点附近不停地徘徊；如果步子迈得太小，下降速度又会太慢，会浪费很多训练时间。

学习速率如果可以改变，就能解决这个问题。我们可以在初始的Iteration中选择比较大的学习速率，之后逐渐减小，这就是Learning Rate Decay.

当然我们这里要增加两个palceholder，一个用来存放训练速率，另一个用来存储当前的步数(literation数) ，并最后在Seesion中通过 feed_dict 传到训练中去。

```python
lr = tf.placeholder(tf.float32)
step = tf.placeholder(tf.int32)
```

Tensorflow 提供Learning rate decay的方法，这个表示训练速率随着Iteration的增加从0.003一指数形式下降到0.0001。

```python
lr = 0.0001 + tf.train.exponetial_decay(0.003, step, 2000, 1/math.e)
```

## 5. Dropout

对付overfitting，我们可以在训练中Dropout掉一定的神经元。在Tensorflow中使用Dropout只需要在相应层中"增加"一个Dropout层。

比如第四层

```python
Y4 = tf.nn.relu(tf.matmul(Y3d, W4) + B4)
Y4d = tf.nn.dropout(Y4, rate = drop_rate)
```

注意在验证的时候，drop rate要设置为0

加上learning rate decay 和 dropout之后的训练sesssion如下

```python
history = {'acc_train':list(),'acc_val':list(),
           'loss_train':list(),'loss_val':list(),
          'learning_rate':list()}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(20000):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={X: batch[0], Y_true: batch[1], step: i, drop_rate: 0.25})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            # Test the Train Model
            feed_dict_train = {X: batch[0], Y_true: batch[1], drop_rate: 0.25}
            feed_dict_val = {X:ch.test_images, Y_true:ch.test_labels, drop_rate: 0}

            matches = tf.equal(tf.argmax(Y,1),tf.argmax(Y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            history['acc_train'].append(sess.run(acc, feed_dict = feed_dict_train))
            history['acc_val'].append(sess.run(acc, feed_dict = feed_dict_val))

            history['loss_train'].append(sess.run(cross_entropy, feed_dict = feed_dict_train))
            history['loss_val'].append(sess.run(cross_entropy, feed_dict = feed_dict_val))
            
            history['learning_rate'].append(sess.run(lr, feed_dict = {step: i}))
            print("Iteration {}:\tlearning_rate={:.6f},\tloss_train={:.6f},\tloss_val={:.6f},\tacc_train={:.6f},\tacc_val={:.6f}"
                  .format(i,history['learning_rate'][-1],
                          history['loss_train'][-1],
                          history['loss_val'][-1],
                          history['acc_train'][-1],
                          history['acc_val'][-1]))
            
            print('
')
        
    saver.save(sess,'models_saving/my_model.ckpt'
```

## 6. 训练效果

可以看到通过dropout 和 learning rate decay 之后，神经网络对MNIST手写数字的识别率已经能提高到98%了，如何进一步提高识别率呢？我们就必须会引入卷积神经网络了。

---

参考资料

[1][https://www.kaggle.com/c/digit-recognizer/data](https://www.kaggle.com/c/digit-recognizer/data)

[2][https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0](https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0)

[3][https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd.git)

[4][https://www.tensorflow.org/api_docs/](https://www.tensorflow.org/api_docs/)

---

相关文章

[Tensorflow入门——单层神经网络识别MNIST手写数字](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)

[AI学习笔记——Tensorflow中的Optimizer](https://steemit.com/tensorflow/@hongtao/ai-tensorflow-optimizer)

[Tensorflow入门——分类问题cross_entropy的选择](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)

[AI学习笔记——Tensorflow入门](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)

[Tensorflow入门——Keras简介和上手](https://steemit.com/cn-stem/@hongtao/tensorflow-keras)

---

同步到我的简书和Steemit

[https://www.jianshu.com/u/bd506afc6fc1](https://www.jianshu.com/u/bd506afc6fc1)

<https://steemit.com/@hongtao>

