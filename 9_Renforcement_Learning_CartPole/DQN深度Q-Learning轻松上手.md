之前的文章简单介绍了深度Q-learning的理论以及Q-learng的实战，这篇文章我们就来实践一下与深度学习相结合的Q-learning——Deep Q-learning。

同样的，为了方便与读者交流，所有的代码都放在了这里：

https://github.com/zht007/tensorflow-practice

### 1. Q-learning与深度学习回顾

在Q-learning中我们需要建立一张Q表，这张Q表记录着每个状态S和A对应的Q值:**Q(S,A)**，根据这张表智能体就可以通过相应的策略(比如epsilon greedy)，来指导行动。

> Q-Learning的过程就是不断跟新Q表的过程：
> $$
> Q[s, a]+=\text { learning_rate } *(\text { td_target }-Q[s, a])
> $$
> 其中：
> $$
> \text { td_target }=R[t+1] + \text {discout_factor}*max(Q[s'])
> $$
> s‘代表下一个状态

深度学习就是建立一个多层的神经网络，通过训练和学习，我们可以用这个神经网络完成**输入特征值**，**输出结果**；以实现分类、预测和感知等功能。

### 2. 深度Q-Learning的优势

Q-learning依赖Q表，对于状态有限的问题，比如之前文章介绍的21点游戏以及格子世界等，能轻松应对。但是对于连续性状态，或者状态特别多的问题，我们需要建立一个超大的Q表，这无疑对计算机的存储器容量的要求提出非常高的要求。即便我们能够建立这个超级庞大的Q表，训练起来也是非常困难。

为了解决这个问题，我们可以用神经网络来替代Q表，输入状态S和A，输出预测的Q值，智能体同样可以根据这个Q值结合相应的策略(比如epsilon greedy)，来指导行动。接着我们通过训练和更新神经网络，达到更新行动策略的目的。

这种用神经网络或者其他函数替代Q表的方法我们叫做**值函数近似(Value Function Approximation)**，在之前的文章中也有所介绍了。

通过与深度学习的结合，我们就把强化学习的问题转换成了类似于**监督学习(Supervised Learning)**的问题了。只不过在监督学习中，我们用**标签**或已知的数据来修正神经网络，在深度Q-learning中，我们用td_target来修正神经网络。

![deep q-learning](http://ww2.sinaimg.cn/large/006tNc79gy1g3bqfvtr6lj318w0tgq8g.jpg)

### 3. Target 和 Prediction 网络

与监督学习中标签永远是固定的，所以神经网络迭代的过程也是稳定的。然而在DQN中，td_target是随着神经网络中参数的更新而更新的，这就意味着神经网络很有可能会出现无法收敛和不稳定的问题。

为了解决这个问题，我们需要暂时固定一个网络，我们叫这个网络为Target网络，这个网络产生的**稳定的"标签"**来更新另外一个神经网络的参数。另一个网络就叫做**Prediction 网络**。

![img](http://ww3.sinaimg.cn/large/006tNc79gy1g3bqgbffkij31050u07ac.jpg)

### 4. 记忆回放(Experience Replay)

Q-learning 还有一个缺点是，智能体每行动一步，Q表就更新了，该状态下的行动，奖励，和下一个状态的结果(s, a, r, s') 就被抛弃了。DQN中使用**记忆库**存储很长一段时间的(s, a, r, s')序列。在训练的过程中随机抽取其中的数据对Prediction 网络进行更新。

由于两个神经网络的结构完全相同，经过若干次迭代后，将Predction 网络的参数复制给Target 网络，如此循环，便完成了DQN的迭代。

![img](http://ww1.sinaimg.cn/large/006tNc79gy1g3bqgnmkyhj30rw0detcc.jpg)

### 5. Keras实践DQN

从头写DQN的确是一件很繁琐的任务，不过开源的keras-rl 库让一切变得非常简单，仅需几步，我们就可以用DQN来玩open gym中的游戏啦。

以下代码均参考[官方github的example](https://github.com/keras-rl/keras-rl/blob/master/examples/dqn_cartpole.py) with MIT License。

#### 5.1 安装 Keras-rl库

pip 安装

```
pip install keras-rl
```

或者从github安装

```
git clone https://github.com/keras-rl/keras-rl.git
cd keras-rl
python setup.py install
```

当然 gym 和 h5py 依赖库也需要安装

```
pip install h5py
pip install gym
```

#### 5.2 初始化gym

import 必要的库

```python
import numpy as np
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
```

初始化gym 环境，这里用经典的CartPole游戏

```python
ENV_NAME = 'CartPole-v0'

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)

nb_actions = env.action_space.n
```

#### 5.3 Keras搭建深度神经网络

Keras搭建神经网络非常简单，就不多讲了，需要注意的是，输入的shape是(1,observation_space)

```python
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())
```

#### 5.4 配置智能体

memory就是之前提到的**记忆库**，epsilon greedy 的策略，Keras自带的Adam优化器。

```python
memory = SequentialMemory(limit=50000, window_length=1)
policy = EpsGreedyQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])
```

#### 5.5 训练和验证

与Keras 一样，我们调用dqn的fit方法进行训练，传入环境，循环步数就可以了，当然我们也可以开启动画，只不过这样会拖慢训练进度。

```python
H = dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)
```

验证的时候我们可以开启动画，看看训练效果

````python
dqn.test(env, nb_episodes=5, visualize=True)
````

![img](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/01/11103833/20160503205408128.gif)

当然我们也可以画出奖励与训练步数的关系图，可以看到当迭代到2000到4000步的时候，我们就可以得到比较好的奖励数值了

![image-20190523173123723](http://ww4.sinaimg.cn/large/006tNc79gy1g3bq9wnrrej30fu0a8gmr.jpg)



最后，由于keras-rl完全兼容open gym的环境，所以如果想玩其他游戏比如Mountain car, 仅需要将 ENV_NAME 改个名字就可以了，至于如何调整参数和神经网络的结构就留给读者自己探索了。

```python
ENV_NAME = 'MountainCar-v0'
```



-------

参考资料

[1] [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)

[2] [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

[3] [Github repo: Reinforcement Learning](https://github.com/dennybritz/reinforcement-learning)

------

相关文章

[强化学习——MC(蒙特卡洛)玩21点扑克游戏](https://steemit.com/cn-stem/@hongtao/mc-21)

[强化学习实战——动态规划(DP)求最优MDP](https://steemit.com/cn-stem/@hongtao/dp-mdp)

[强化学习——强化学习的算法分类](https://steemit.com/ai/@hongtao/7atbof)

[强化学习——重拾强化学习的核心概念](https://steemit.com/ai/@hongtao/2bqdkd)

[AI学习笔记——Sarsa算法](https://steemit.com/ai/@hongtao/ai-sarsa)

[AI学习笔记——Q Learning](https://steemit.com/ai/@hongtao/ai-q-learning)

[AI学习笔记——动态规划(Dynamic Programming)解决MDP(1)](https://steemit.com/ai/@hongtao/ai-dynamic-programming-mdp-1)

[AI学习笔记——动态规划(Dynamic Programming)解决MDP(2)](https://steemit.com/ai/@hongtao/ai-dynamic-programming-mdp-2)

[AI学习笔记——MDP(Markov Decision Processes马可夫决策过程)简介](https://steemit.com/ai/@hongtao/ai-mdp-markov-decision-processes)

[AI学习笔记——求解最优MDP](https://steemit.com/ai/@hongtao/ai-mdp)

------

同步到我的简书
https://www.jianshu.com/u/bd506afc6fc1