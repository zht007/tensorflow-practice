![img](http://ww4.sinaimg.cn/large/006tNc79gy1g4stgr2nb7j30rs0kt45l.jpg)

*Image from [unsplash.com](https://unsplash.com/photos/i3mcVZQObcU) by Ferdinand Stöhr*

前文我们讲了如何用Q-learning 和 SARSA 玩推小车上山的游戏，这篇文章我们探讨一下如何完成Carpole平衡杆的游戏。

同样的，为了方便与读者交流，所有的代码都放在了这里：

https://github.com/zht007/tensorflow-practice

### 1. 环境分析

关于cartPole 游戏的介绍参见之前[这篇文章](https://steemit.com/cn-stem/@hongtao/dqn-q-learning)，这里就不赘述了。通过阅读官方文档，Open AI 的 [CartPole v0](https://github.com/openai/gym/wiki/CartPole-v0) 可以发现，与[MountainCar-v0](https://github.com/openai/gym/wiki/MountainCar-v0) 最大的区别是，CartPole 的状态有四个维度，分别是位置，速度，夹角和角速度。其中，速度和角速度的范围是正负无穷大。我们知道Q-learning 和 SARSA 都依赖有限的表示非连续状态的策略(Q-表)，如何将无限连续的状态分割成有限不限连续的状态呢？

这里我们可以使用在神经网络中被曾被广泛应用的 sigmoid 函数，该函数可以将无限的范围投射在0到1之间。所以我们先建立这个 sigmoid 帮助函数。

```python
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
```

### 2. 建立Q-表

与MountainCar 类似需要将连续的状态切割成离散的状态，不同的是速度和角速度需要用sigmoid 函数投射在有限的范围内。

```python
DISCRETE_OS_SIZE = [Q_TABLE_LEN] * (len(env.observation_space.high))


observation_high = np.array([env.observation_space.high[0],
                    Q_TABLE_LEN*sigmoid(env.observation_space.high[1]),
                    env.observation_space.high[2],
                    Q_TABLE_LEN*sigmoid(env.observation_space.high[3])])

observation_low = np.array([env.observation_space.low[0],
                    Q_TABLE_LEN*sigmoid(env.observation_space.low[1]),
                    env.observation_space.low[2],
                    Q_TABLE_LEN*sigmoid(env.observation_space.low[3])])

discrete_os_win_size = (observation_high - observation_low) / DISCRETE_OS_SIZE
```

*Code from [github repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/1_q_learning_python_mountain_car.ipynb) with MIT license* 

值得注意的是，由于Q-表的维度比较高，这里将其参数直接设置为0，否则随机产生150 * 150 *150 *2 个数需要花费很长时间。另外 Q_TABLE_LEN 我设置的是150 (大约占用6G的内存)，过大的Q-表长度会导致内存溢出。

```python
q_table = np.zeros((DISCRETE_OS_SIZE + [env.action_space.n]))
```



### 3. Q - Learning 和 SARSA 

后面的代码与 MountainCar 几乎一模一样，这里就不赘述了，可参考[前文](https://steemit.com/cn-stem/@hongtao/q-learning-mountaincar)。可以发现两者区别不大，均很好地完成了任务。

![image-20190708154308650](http://ww2.sinaimg.cn/large/006tNc79gy1g4stak2o2mj312m0ek78t.jpg)

理论上来说，SARSA lambda 也是可以使用的，但是由于智能体每走一步均需要更新整个Q表，然而该表又实在太大实践起来计算量非常之巨大，感兴趣的读者可自行尝试。

---

参考资料

[1] [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)

[2] [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

[3] [Github repo: Reinforcement Learning](https://github.com/dennybritz/reinforcement-learning)

-----

相关文章

[强化学习—— SARSA 和 SARSA lambda 玩 MountainCar 爬坡上山](https://steemit.com/cn-stem/@hongtao/sarsa-sarsa-lambda-mountaincar)

[强化学习—— Q-Learning 玩 MountainCar 爬坡上山](https://steemit.com/cn-stem/@hongtao/q-learning-mountaincar)

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

----

同步到我的简书 https://www.jianshu.com/u/bd506afc6fc1