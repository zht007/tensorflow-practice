在之前的文章里面，我介绍了如何用Tensorflow或者Keras搭建神经网络，但是还没有介绍什么是神经网络。本文就用简单通俗的语言介绍一下，希望是抛砖引玉。

### 1.神经网络（Neural Network(NN)）

一个典型的**神经网络**如下图
![img](https://ws1.sinaimg.cn/large/006tNc79gy1g2cr7y2zwej30c8086q5e.jpg)

其最基本的神经元是由一个线性函数和一个非线性的激活函数组成：
![img](https://ws4.sinaimg.cn/large/006tNc79gy1g2cr847h7kj30cj051wfy.jpg)


这个线性函数与之前[线性回归](https://steemit.com/cn/@hongtao/ai-linear-regression)是一样的，而激活函数可以理解为将输出的结果进行调控，比如使其保证在0和1之间。

与线性回归一样，神经网络实际上就是要训练找到合适的w 和 b。与线性回归一样，使用梯度下降(Grident Dscent)法，即可得到最优 的w和b。

非线性的激活函数有很多类，如图：
![img](https://ws2.sinaimg.cn/large/006tNc79gy1g2cr8bb4htj30g309843s.jpg)


Sigmoid 是早期比较流行的，不过现在用的最多的是ReLu,为什么简单的Relu能够比看起来更加合理的Sigmoid 更加有效，并不是这篇笔记要探讨的话题。

至于为什么要用激活函数，我想也很好理解，如果整个神经网络全是线性的话，那么无论这个网络有多复杂，最终都只是一个线性的，然而我们这个世界上的事物用线性模型就都能解释吗？显然不行。

### 2.深度神经网络（Deep Neural Network (DNN)）

深度神经网络实际上就是将神经网络进行叠加，而中间层叫做隐藏层（Hiden layer）, 隐藏层能够分辨出浅层神经网络无法分辨的细节。
![img](https://ws1.sinaimg.cn/large/006tNc79gy1g2cr8m9s8kj30hs0ay43e.jpg)

### 3.前向传播和反向传播（Forward and Backward propagation）

前向传播其实很简单，就是如何堆砌这个神经网络，多少个Feature 输入，多少层神经网络，每层多少个神经元，每一层用什么激活函数。

最困难的是反向传播，类似于线性回归一样，我们的目的是要用合适的参数（W和b）使这个网络，或者说整个模型预测的值最接近真实的数值，换句话说就是预测值与真实值的差距最小。这个求这个差值的函数我们叫**代价函数(Cost Function)**, 而反向传播就是通过预测结果，向前倒推每一层W和b的导数。通过这个导数我们就可以用梯度下降的方法训练出代价函数最小的W和b值。

反向传播涉及到了微分，和偏微分（偏导）递归等数学原理，虽然也不难，但是也并不在本文的讨论范围之内。不过好消息是在现在流行的深度学习工具中，比如在Tensorflow中, 我们只需要关心如何搭建这个网络（前向传播），工具会自动通过反向传播计算最优解，反向传播优化器的选择可以参见这篇文章。

其实听起来高大上的NN和DNN是不是很简单。

* * *
相关文章

[Tensorflow入门———Optimizer(优化器)简介](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483697&idx=1&sn=df3943301226106817fe217c93767a78&chksm=fd828ea2caf507b42c65543a43ff8e5479b800ce84f2d2086e5c38bcc2406b342c7fa74fce6c#rd)

[Tensorflow入门——分类问题cross_entropy的选择](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483693&idx=1&sn=25971f9727b486ceb7470b215a780ada&chksm=fd828ebecaf507a8173a79fadea560d6ab30850aaf0826d65e336fc2726bd99a2e94a8f94b3d#rd)

[快速上手——解决过拟合(overfitting)的问题](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483688&idx=1&sn=60c97e12a69afd11d421f49de482d5d4&chksm=fd828ebbcaf507ade3d09cc7c49c1202e871d5d5afe9b57883d5c89dbae7075b0fbabd866e42&scene=21#wechat_redirect)
[快速上手——用Tensorflow处理分类问题](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483683&idx=1&sn=c1d8b401c05640398b195b1e76676c88&chksm=fd828eb0caf507a6ea4ebee4e44ca952bca81de820ef33235d76241ec42200cf8269edf0ab16&scene=21#wechat_redirect)
[免费使用Google的GPU和TPU来训练你的模型](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483668&idx=1&sn=c0522edd98f3ada940eb4b655cf1215b&chksm=fd828e87caf50791fa18af9c8d395e97096afdb94f06979a6635839c5a7493355faf4e2bf9e2&scene=21#wechat_redirect)
[Tensorflow入门——Keras处理分类问题](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483673&idx=1&sn=ac1204891e1e99c13930d07cdec9c307&chksm=fd828e8acaf5079cacb31cd23223615cc5e67049998ff1aa785d704a7ec549b17b71a1dcc448&scene=21#wechat_redirect)
[Tensorflow入门——Keras简介和上手](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483663&idx=1&sn=dcdb7b988c6243548fb32610498f60d1&chksm=fd828e9ccaf5078a285270ebfe529c887ed9c68318e2a9b120cc14c4e9e34d7b7816da3bf388&scene=21#wechat_redirect)
[Tensorflow入门——两分钟完成“线性回归“训练](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483659&idx=1&sn=eb15854d3360e50e7440914536146f89&chksm=fd828e98caf5078e8b2337711c96084aa01067d3ef0f8a971f1e29d34ea63f54709d130df450&scene=21#wechat_redirect)
[Tensorflow入门——我的第一个tf程序](http://mp.weixin.qq.com/s?__biz=MzU4NTg4MjM2NA==&mid=2247483653&idx=1&sn=2c669742d188c3078f2461ca22718781&chksm=fd828e96caf5078031d20ade9e027220f29c55a29ca2369fec0d26bec4f6752c3a9ff98e3952&scene=21#wechat_redirect)

欢迎扫描二维码关注我的微信公众号“tensorflow机器学习”，一起学习，共同进步
