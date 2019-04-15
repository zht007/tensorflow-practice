[上一篇文章](https://steemit.com/cn-stem/@hongtao/ai-you-only-look-once-yolo)介绍了YOLO和目标识别算法，大家是不是迫不及待跃跃欲试了。先不要着急，YOLO本身原理虽然简单，但是实现起来也不容易。YOLO原生是用C语言写的，放在无法直接用Python实现。

当然我们完全可以根据其原理搭建一个YOLO模型，但是对于初学者而言，完全没有必要再发明轮子嘛。github上的大神们已经用各种流行的框架实现了YOLO，今天我们就来介绍一个通过Keras实现YOLO的项目。通过这个项目，我们只需要三步，就能实现视频和图片的目标识别。

项目地址在这里

<https://github.com/qqwweee/keras-yolo3>

## 1. 官网下载YOLO模型参数

当然，你可以自己训练模型的参数，但是普通人恐怕没这个硬件条件，我们只是想体验一下YOLO的识别效果嘛，模型参数直接去[YOLO官网](http://pjreddie.com/darknet/yolo/)下载就好了。

首先git clone 整个项目，不会git的朋友可以直接下载zip压缩包，当然打开终端cd到项目根目录，命令行一键可下载模型参数

```python
wget https://pjreddie.com/media/files/yolov3.weights
```

## 2. 模型参数转换

注意模型**参数权重**和模型参数的**配置文件**一定要匹配，项目中已经有两个配置文件yolov3.cfg和yolov3-tiny.cfg，所以只需要下载对应的权重文件就可以了。[YOLO官网](http://pjreddie.com/darknet/yolo/)也提供了配置文件和相应的参数权重。

运行项目文件下的convert.py完成参数转换，将模型转换成Keras的h5文件。

```python
python convert.py yolov3.cfg yolov3.weights model_data/yolo.h5
```

## 3. 模型使用

使用模型仅需要运行yolo_video.py这个文件就可以了。识别图片仅需要运行这一行命令

```
python yolo_video.py --image
```

终端会提示输入图片文件的filename

```
Input image filename:
```

将你要检测的图片文件放在项目文件夹下，输入文件名，目标识别就开始啦。

我们找一张图片试试效果吧

![image-20190412161543361](https://ws3.sinaimg.cn/large/006tNc79gy1g209brpjimj30sy0xihdt.jpg)

*image source from [unsplash](https://images.unsplash.com/photo-1554772593-cc0206eee02b?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=668&q=80) by Chander Mohan*

当然也可以对视频文件进行目标识别，由于我的笔记本运行起来比较慢，这里就不做演示了，yolo_video.py的使用说明如下

```
usage: yolo_video.py [-h] [--model MODEL] [--anchors ANCHORS]
                     [--classes CLASSES] [--gpu_num GPU_NUM] [--image]
                     [--input] [--output]

positional arguments:
  --input        Video input path
  --output       Video output path

optional arguments:
  -h, --help         show this help message and exit
  --model MODEL      path to model weight file, default model_data/yolo.h5
  --anchors ANCHORS  path to anchor definitions, default
                     model_data/yolo_anchors.txt
  --classes CLASSES  path to class definitions, default
                     model_data/coco_classes.txt
  --gpu_num GPU_NUM  Number of GPU to use, default 1
  --image            Image detection mode, will ignore all positional 
```

*code from [github](https://github.com/qqwweee/keras-yolo3) with MIT License*

## 尾巴

简单三步我们在自己的电脑上就实现了YOLO模型，感兴趣的朋友可以去[项目地址](https://github.com/qqwweee/keras-yolo3)下载并阅读源代码。

----

相关文章

[AI学习笔记——精准识别You Only Look Once(YOLO)](https://steemit.com/cn-stem/@hongtao/ai-you-only-look-once-yolo)

[Tensorflow入门——单层神经网络识别MNIST手写数字](https://steemit.com/cn-stem/@hongtao/tensorflow-mnist)

[Tensorflow入门——多层神经网络MNIST手写数字识别](https://steemit.com/cn-stem/@hongtao/6qe2nw-tensorflow-mnist)

[AI学习笔记——Tensorflow中的Optimizer](https://steemit.com/tensorflow/@hongtao/ai-tensorflow-optimizer)

[Tensorflow入门——分类问题cross_entropy的选择](https://steemit.com/cn-stem/@hongtao/tensorflow-crossentropy-how-to-choose-crossentropy-loss-in-tensorflow-for-classification)

[AI学习笔记——Tensorflow入门](https://steemit.com/cn-stem/@hongtao/ai-tensorflow)

[Tensorflow入门——Keras简介和上手](https://steemit.com/cn-stem/@hongtao/tensorflow-keras)

----

------

同步到我的简书

<https://www.jianshu.com/u/bd506afc6fc1>