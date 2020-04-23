# 免费使用Google的GPU和TPU来训练你的模型

万事开头难，对于机器学习初学者来说，最困难的可能是如何在计算机中搭建机器学习所需要的环境，特别是[如何配置GPU问题](https://steemit.com/cn/@hongtao/mac-tensorflow-gpu)，让很多老鸟都颇为头疼。

好消息是Google爸爸已经替大家把这些问题都解决了，现在只需要有一台能(科学)上网电脑，一个浏览器就能可以开启机器学习之路了，Google甚至把昂贵的GPU和TPU都开放出来了，还等什么，赶紧上车吧。

## 1.Colab简介

[Colab](https://colab.research.google.com)是Google开发的一个完全基于云端的Jupyter Notebook开发环境，使用者不需要做任何设置就可以在上面运行代码。当然也支持从本地和Github导入数据和文件。

![](https://ws4.sinaimg.cn/large/006tKfTcgy1g19i4x2y8xj31bv0u0e3g.jpg)

## 2. Colab保存文件

在Colab中，可以完全像在本地使用Jupyter Notebook，新建，导入，运行Jupyter Notebook.

但是要注意的是如果你想保存文件最好"Save a copy to Google Drive"，否则关掉浏览器数所有数据都会丢失掉的。当然你也可以下载".py"或者".ipnb"文件。

这些选项都在“File”中可以找到。

![](https://ws3.sinaimg.cn/large/006tKfTcgy1g19igwowd0j30zz0u0k0l.jpg)

## 3. Colab实践

我们试一下从Gitbut中导入之前文章中用到的Jupyter Notebook 文件吧。

点击"File ——> Upload Notebook" 选择Github选项并输入Github地址就能找到我们之前的Jupyter Notbook文件啦。

![](https://ws1.sinaimg.cn/large/006tKfTcgy1g19ilp46irj318b0u0wio.jpg)



导入之后，你就可以向在本地运行Jupyter Notebook一样在云端训练你的模型啦。由于训练是在云端进行，本地电脑该干嘛还可以继续干嘛，不用担心功CPU、内存、GPU占用率过高的问题啦。

## 4. 使用GPU和TPU

如果模型太复杂或者数据量太大，GPU和TPU([Tensor processing unit ](https://en.wikipedia.org/wiki/Tensor_processing_unit))可以大大地提高运算和训练速度。

选择GPU和TPU对训练进行加速也非常简单，只需要点击"Runtime —> Change Runtime Type"就可以啦。

![](https://ws3.sinaimg.cn/large/006tKfTcgy1g19ix0fglij30o00ha75i.jpg)



高端的GPU和TPU非常昂贵，但是Google爸爸慷慨得开放给了开发者，这个必须给Google点个赞。

----

以上图片均来自于[Colab](https://colab.research.google.com)官网截图

------

同步到我的简书和Steemit
<https://www.jianshu.com/u/bd506afc6fc1>

<https://steemit.com/@hongtao>

