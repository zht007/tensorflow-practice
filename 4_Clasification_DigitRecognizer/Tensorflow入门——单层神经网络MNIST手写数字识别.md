手写数字识别是一个非常经典的机器学习项目，这篇文章，我们就通过[Kaggle上这个经典项目](https://www.kaggle.com/c/digit-recognizer)，学习如何用Tensorflow和Keras用最简单的单层神经网络，来识别手写数字。

同样的，为了方便与读者交流，所有的代码都放在了这里：

#### Repository:

<https://github.com/zht007/tensorflow-practice>



## 1. 数据下载和预处理

在[Kaggle的项目](https://www.kaggle.com/c/digit-recognizer/data)页面可以下载两个csv文件，"train.csv"包含数据和标签，"test.csv"仅包含带验证数据。你可以用train.csv训练自己的模型，然后再用这个模型识别test.csv中的手写数字，并将其分类，最后将结果上传至[Kaggle的项目](https://www.kaggle.com/c/digit-recognizer/data)，查看正确率和全球排名。

通过pands的read_csv方法读取csv文件，分离数据和标签，并分用scikit-learn 中的train_test_split,分出训练集和验证集。

```python
labeled_images = pd.read_csv('train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, test_size=0.01)
```

## 2. Batch training 的帮助函数

这个部分虽然比较难，但是不是这篇文章的重点，就此略过，主要作用是从训练数据集中顺序取出指定数量的batch，在Session中给模型训练。帮助函数处理之后数据的shape为[batch_size, 28,28,1]

帮助函数还有一个作用，就是将标签onehot encode。onehot-encoded 的标签shape为[batch_size, 10]。

## 3. 创建模型

单层神经网络，神经元个数就等于输出的类别的个数，手写数字分成0到9，一共10个类别，神经元个数就是10。

神经网络是全链接的，我们需要把输入的28*28个像素的二维图片，拆解拼凑成一个784个像素点的一维向量。此时输入的feature数就是784。

输入的feature数和输出的类别数共同决定了权重W和偏移b的shape

初始化权重W1和偏移B1：

```python
W1 = tf.Variable(tf.truncated_normal([784, 10], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.ones([10])/10)
```



单层神经网络通过softmax的激活，就能得到最终的结果

```
XX = tf.reshape(X,[-1,784])
Ylogits = tf.matmul(XX, W1) + B1
Y = tf.nn.softmax(Ylogits)
```

Cross_entropy 可以直接通过公式计算

```python
cross_entropy = -tf.reduce_mean(Y_true * tf.log(Y)) * 1000.0 
```

也可以用tensorflow中自带的，如何选择我在[前面的文章](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)中已经介绍过了。

```python
cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels = Y_true, logits = Ylogits)
```

Optimizer 可以选择基本的GradientDescent也可以选择Adam

```
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)
```

## 4. 模型训练

将batch中的数据通过Feed_dict载入数据，剩下的就交给Tensorflow吧，注意，为了记录loss 和 Accuracy的变化，我创建了history这个字典，记录每100个Iteration它们的数字变化。

```python
history = {'acc_train':list(),'acc_val':list(),
           'loss_train':list(),'loss_val':list()}
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(30000):
        batch = ch.next_batch(100)
        sess.run(train, feed_dict={X: batch[0], Y_true: batch[1]})
        
        # PRINT OUT A MESSAGE EVERY 100 STEPS
        if i%100 == 0:
            
            # Test the Train Model
            feed_dict_train = {X: batch[0], Y_true: batch[1]}
            feed_dict_val = {X:ch.test_images, Y_true:ch.test_labels}

            matches = tf.equal(tf.argmax(Y,1),tf.argmax(Y_true,1))
            acc = tf.reduce_mean(tf.cast(matches,tf.float32))
            history['acc_train'].append(sess.run(acc, feed_dict = feed_dict_train))
            history['acc_val'].append(sess.run(acc, feed_dict = feed_dict_val))

            history['loss_train'].append(sess.run(cross_entropy, feed_dict = feed_dict_train))
            history['loss_val'].append(sess.run(cross_entropy, feed_dict = feed_dict_val))
            
            print("Iteration {}:\tloss_train={:.6f}:\tloss_val={:.6f}:\tacc_train={:.6f}:\tacc_val={:.6f}"
                  .format(i,history['loss_train'][-1],history['loss_val'][-1],history['acc_train'][-1],history['acc_val'][-1]))
            
            print('\n')
```

## 4.查看训练结果

可以看到，即便只有一层神经网络，我们也达到了将近90%的Accuracy.

![image-20190329150857000](https://ws1.sinaimg.cn/large/006tKfTcgy1g1k2gbg2irj30mc0j6q5p.jpg)

## 5. 用Keras试试看

Keras就更加简单了，Model两行代码搞定了。

```python
model = models.Sequential()
model.add(layers.Dense(units=10, activation='softmax',input_shape=(784,)))
```

## 6.预测测试集数据并上传Kaggle

我们在训练的最后已经将tensorflow的模型保存起来了

```python
 saver.save(sess,'models_saving/my_model.ckpt')
```

预测的时候取出来就行了

```python
unlabeled_images_test = pd.read_csv('test.csv')

with tf.Session() as sess:
    
    # Restore the model
    saver.restore(sess, 'models_saving/my_model.ckpt')
    # Fetch Back Results
    label = sess.run(Y, feed_dict={X:X_unlabeled})
```



最后，按照Kanggle提供模板格式，将结果转换成csv文件，上传服务器，就可以看到训练成果啦。

详细过程请参见jupyter notbook中的代码和注释。

------

参考资料

<https://www.kaggle.com/c/digit-recognizer/data>

<https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0>

<https://www.tensorflow.org/api_docs/>

----

同步到我的简书和Steemit

<https://steemit.com/@hongtao>

<https://www.jianshu.com/u/bd506afc6fc1>

