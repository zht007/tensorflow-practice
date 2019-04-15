# Tensorflow入门——Keras处理分类问题，Classification with Keras

Tensorflow 和 Keras 除了能处理[前一篇](https://busy.org/@hongtao/tensorflow-keras)文章提到的回归(Regression，拟合&预测)的问题之外，当然也可以处理分类(Classification)问题。

这篇文章我们就介绍一下如何用Keras快速搭建一个线性分类器，通过分析病人的生理数据来判断这个人是否患有糖尿病。

同样的，为了方便与读者交流，所有的源代码都放在了这里：

https://github.com/zht007/tensorflow-practice/

### 1. 数据的导入

数据的csv文件已经放在了项目目录中，也可以去[Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)下载。

### 1. 数据的导入

数据的csv文件已经放在了项目目录中，也可以去[Kaggle](https://www.kaggle.com/uciml/pima-indians-diabetes-database)下载。

![](https://ws3.sinaimg.cn/large/006tKfTcgy1g15est7xhdj31540f80v5.jpg)

### 2.数据预处理

#### 2.1 Normalization(标准化)数据

标准化数据可以用sklearn的工具，但这里就直接计算了。要注意的是，这里并没有标准化年龄，年龄问题需要特殊处理。

```python
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
       'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
```

#### 2.2 年龄分段

对于包含年龄这样的数据，通常需要对数据按年龄段分类，我们先看看数据中病人的年龄分布。

![](https://ws2.sinaimg.cn/large/006tKfTcgy1g15f5pmw97j30qu0lwac5.jpg)

可以通过panda自带的cut函数对年龄进行分段，我们这里将年龄分成0-30，30-50，50-70，70-100四段，分别标记为0，1，2，3

```python
bins = [0,30,50,70,100]
labels =[0,1,2,3]
diabetes["Age_buckets"] = pd.cut(diabetes["Age"],bins=bins, labels=labels, include_lowest=True)
```

#### 2.3 训练和测试集分离

这一步不用多说，还是用sklearn.model_selection 的 train_test_split工具进行处理。

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=101)
```

### 3. 用Keras搭建线性分类器

模型的搭建与[前一篇](https://busy.org/@hongtao/tensorflow-keras)文章中的线性回归模型没有太大区别，唯一不同的是线性分类器要输出两个结果(是否患病)，所以需要两个神经元Unit =2，而且还需要加一个"softmax"的激活函数。

"softmax"的激活函数的作用是将最后的分类结果转换成概率。所以我们最后得到的不是一个确定的类别，而是模型预测属于每个类别的概率(是否患病的概率)。

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.utils import to_categorical

model = Sequential()
model.add(Dense(2,input_shape = (X_train.shape[1],),activation = 'softmax'))
```

需要注意的是标签y需要进行One Hot encoding的转换，实际上是将一元数据转换成二元数据(Binary)的"One Hot"数据。比如原始标签用"[1]"和"[0]"这样的一元标签来标记"是"“否”患病，转换之后是否患病用"[1 , 0]"和"[0 , 1]"这样的二元标签来标记。

```python
y_binary_train= to_categorical(y_train)
y_binary_test = to_categorical(y_test)
```

同样可以选用SGD的优化器，但是要注意的是，在Compile的时候损失函数要选择"categorical_crossentropy"。Coross Entropy是在分类问题中计算预测结果与实际结果的“差距”。

```python
sgd = SGD(0.005)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics=['accuracy'])
```

### 4. 分类器的训练

训练的时候可以直接将测试数据带入，以方便评估训练效果。

```python
H = model.fit(X_train, y_binary_train, validation_data=(X_test, y_binary_test),epochs = 500)
```

### 5. 训练效果验证

训练效果可以直接调用history查看损失函数和准确率的变化轨迹。从下图可看出线性分类器的分类效果还不错。

![](https://ws1.sinaimg.cn/large/006tKfTcgy1g15g21e2p5j30oy0jc0v8.jpg)

### 6. 深度神经网络

深度神经网络的搭建也很简单，直接在原有的线性分类器上直接“叠蛋糕”就可以了。这里我搭建了一个20x10的三层全连接的神经网络，优化器选用adam

```python
model = Sequential()
model.add(Dense(20,input_shape = (X_train.shape[1],), activation = 'relu'))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))

adam = Adam(0.01)
```

可以看到，虽然精确度比采用线性分类器稍高，但是在200个epoch之后，明显出现过拟合(Over fitting)的现象。

![](https://ws4.sinaimg.cn/large/006tKfTcgy1g15g8670u8j30og0jc43p.jpg)

### 7. 用模型进行预测

同样的我们可以用训练得到的模型对验证数据进行预测(分类)，这里需要注意的是，正如之前我们提到的，模型得到的结果并不是类别，而是属于每个类别的概率。我们最后需要用np.argmax得到概率最大的那个类别。

```python
import numpy as np
y_pred_softmax = model.predict(X_test)
y_pred = np.argmax(y_pred_softmax, axis=1)
```

------

同步到我的简书和Steemit
<https://www.jianshu.com/u/bd506afc6fc1>

<https://steemit.com/@hongtao>