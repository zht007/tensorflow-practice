![books in shelf](https://images.unsplash.com/photo-1576289853729-0445e19a7e16?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1000&q=80)

*image from [unsplash.com](https://unsplash.com/photos/kdUgoghLDAg) by [Tara Hegerty](https://unsplash.com/@tarahegerty)*

[之前的文章](https://www.jianshu.com/p/b0e98ee80a49)介绍过如何在 Keras 中快速调用 Tensorboard 这个可视化工具，这篇文章我们再深入探讨一下 Tensorboard 在T Tensorflow 2.0 下如何使用。

*本文源代码请关注微信公众号(二维码见文末)获取*

### 1. Tensorboard 工作原理

简单来说，tensorboard 就是通过监听定目录下的 log 文件然后在 Web 端将 log 文件中需要监听的变量可视化出来的过程。

所以，使用 Tensorflow 大致分为以下三步：

> 1. 创建监听目录 `logdir`
> 2. 创建 `summary_writer` 对象写入 `logdir`
> 3. 将数据写入到 `summary_writer` 中

### 2.  Tensorboard 监听变量

我们用 Tensorboard 最多的就是实时观察 loss 和 accuracy 在训练过程中的变化，下面就以 MNIST 识别手写数字为例，介绍如何使用 Tensorboard.

首先，创建监听目录和 `summary_witer` 目录名字加上时间戳，以避免混淆。

```python
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time
summary_writer = tf.summary.create_file_writer(log_dir)
```

然后，训练过程中将数据写入 `summary_witer` 中。这里我们将每个 epoch 中的 loss 和 accuracy 写入，仅需要在 epoch 的循环中加入以下代码：

```python
  with summary_writer.as_default():
    tf.summary.scalar("epoch_loss", loss.numpy(), step=epoch)
    tf.summary.scalar("epoch_test_acc", accuracy, step=epoch)
```

其他部分关于模型的建立，训练和测试，参考[之前的文章](https://www.jianshu.com/p/e68172ba8c91)，这里就不赘述了。

当然，我们也可以监听每个 epoch 下，每(若干)个step，仅需要在 step 的循环中加入如下代码，注意，为了避免重复，监听变量的名字也加上了 epoch 的数字：

```python
      with summary_writer.as_default():
        tf.summary.scalar("loss epoch: "+str(epoch), loss.numpy(), step = step)
        tf.summary.scalar("test_acc epoch: "+str(epoch), accuracy, step = step)
```

### 3. 查看 Tensorboard

在 Windows 的 CMD 或 Mac/linux 的terminal中在项目目录下输入以下命令: 可得到一个地址和端口。

```
tensorboard --logdir logs
```

其中 "logs" 是我们指定的监听目录的**父目录**，前面我们将带有时间戳的子目录放在这个目录下，该目录下所有的子目录都可以被监听。

将地址和端口复制到浏览器中即可打开 Tensorboard. Tensorboard 默认没几秒钟就会刷新一次更新数据，当然你也可以手动刷新。

如果你是使用 Colab 的notebooks，现在 Tensorflow 2.0 也支持直接在 notebook 中查看 tensorboard 了。

> 1. 在调用 Tensorflow 2.0 的时候 同时调用 tensorboard，仅需一行代码 `%load_ext tensorboard`
> 2. 创建完成 `summary_writer` 后 同样一行代码显示 tensorboard `%tensorboard --logdir logs`

下图就是 tensorboard 中实时显示每个 epoch loss 和 accuracy 的变化，以及每个 epoch 中 每个 step 下 loss 和 accuracy 的变化。

![image-20191214214034062](https://tva1.sinaimg.cn/large/006tNbRwgy1g9wyvt4hfej30x00u0af7.jpg)

-----

相关文章

[Tensorflow2.0-数据加载和预处理](https://www.jianshu.com/p/b796823ad32c)

[Tensorflow 2.0 快速入门 —— 引入Keras 自定义模型](https://www.jianshu.com/p/e68172ba8c91)

[Tensorflow 2.0 快速入门 —— 自动求导与线性回归](https://www.jianshu.com/p/c44705808f7e)

[Tensorflow 2.0 轻松实现迁移学习](https://www.jianshu.com/p/54aa43935c2b)

[Tensorflow入门——Eager模式像原生Python一样简洁优雅](https://www.jianshu.com/p/883addf4a1b2)

[Tensorflow 2.0 —— 与 Keras 的深度融合](https://www.jianshu.com/p/9a5ae5d92dba)

------

首发[steemit](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2F%40hongtao)

欢迎扫描二维码关注我的微信公众号“tensorflow机器学习”，一起学习，共同进步

![img](https://upload-images.jianshu.io/upload_images/10816620-67b5369ba3a3a00d.png?imageMogr2/auto-orient/strip|imageView2/2/w/258/format/webp)