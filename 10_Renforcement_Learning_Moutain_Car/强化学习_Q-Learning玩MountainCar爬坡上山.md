之前的文章结合理论和实践熟悉了 Q-Learning 的经典算法，这篇文章我们基于 Open AI 的经典 MountainCar 环境。用 python 代码实现 Q-Learning  算法，完成小车爬坡上山的挑战。

同样的，为了方便与读者交流，所有的代码都放在了这里：

https://github.com/zht007/tensorflow-practice

### 1. Gym 环境初始化

要熟悉 MountainCar-v0 的环境请参见官网以及[官方的 github repo](https://github.com/openai/gym/wiki/MountainCar-v0). 

MountainCar-v0 的**环境状态**是由 其位置和速度决定的。**行为 Action** 有三个，向左 (0)，向右 (2)，无(1) 推车。**奖励**: 除了超过目的地 (位置为 0.5)， 其余地方的奖励均为 "-1"

初始化 gym 环境的代码如下

```python
env = gym.make("MountainCar-v0")
env.reset
```

当然强化学习中的参数不要忘了初始化

```python
LEARNING_RATE = 0.5
DISCOUNT = 0.95
EPISODES = 10000
SHOW_EVERY = 500
```

*Code from [Github Repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/1_q_learning_python_mountain_car.ipynb) with MIT lisence*

### 2. Q-Table 的建立

Q表是用来指导每个状态的行动，由于该环境状态是连续的，我们需要将连续的状态分割成若干个离散的状态。状态的个数即为 Q 表的size。这里我们将Q表长度设为20，建立一个 20 x 20 x 3 的Q表。

```python
DISCRETE_OS_SIZE = [Q_TABLE_LEN] * len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

q_table = np.random.uniform(low=0, high=1,
                            size=(DISCRETE_OS_SIZE + [env.action_space.n]))
```

Code from [Github Repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/1_q_learning_python_mountain_car.ipynb) with MIT lisence

另外，我们采用 epsilon-greedy 的策略，epsilon 采用衰减的方式，一开始为1最后衰减为0，也就是说智能体一开始**勇敢探索**，接下来**贪婪行动**获取最大奖励。

```python
epsilon = 1  # not a constant, qoing to be decayed
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)
```

Code from [Github Repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/1_q_learning_python_mountain_car.ipynb) with MIT lisence

### 3. 帮助函数

将环境"离散"化，以适应离散的Q表

```python
def get_discrete_state (state):
    discrete_state = (state - env.observation_space.low) // discrete_os_win_size
    return tuple(discrete_state.astype(int))
```

 epsilon-greedy 策略帮助函数

```python
def take_epilon_greedy_action(state, epsilon):
    discrete_state = get_discrete_state(state)
    if np.random.random() < epsilon:
        action = np.random.randint(0,env.action_space.n)
    else:
        action = np.argmax(q_table[discrete_state])
    return action

```

Code from [Github Repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/1_q_learning_python_mountain_car.ipynb) with MIT license

### 4. 训练智能体

Q-learning属于单步Temporal Difference (时间差分TD(0))算法，其通用的更新公式为
![Q[s, a]+=\text { learning_rate } *\left(\text {td_target}-Q[s, a]\right)](https://math.jianshu.com/math?formula=Q%5Bs%2C%20a%5D%2B%3D%5Ctext%20%7B%20learning_rate%20%7D%20*%5Cleft(%5Ctext%20%7Btd_target%7D-Q%5Bs%2C%20a%5D%5Cright))
其中 td_target - Q[s,a] 部分又叫做 TD Error.

**Q-learning:**
![\text { td_target }=R[t+1] + \text {discout_factor}*max(Q[s'])](https://math.jianshu.com/math?formula=%5Ctext%20%7B%20td_target%20%7D%3DR%5Bt%2B1%5D%20%2B%20%5Ctext%20%7Bdiscout_factor%7D*max(Q%5Bs%27%5D))

核心代码如下:

```python
for episode in range(EPISODES):
    # initiate reward every episode
    ep_reward = 0
    state = env.reset()
    done = False
    while not done:
        action = take_epilon_greedy_action(state, epsilon)

        next_state, reward, done, _ = env.step(action)

        ep_reward += reward
        if not done:

            td_target = reward + DISCOUNT * np.max(q_table[get_discrete_state(next_state)])

            q_table[get_discrete_state(state)][action] += LEARNING_RATE * (td_target - q_table[get_discrete_state(state)][action])

        elif next_state[0] >= 0.5:
            # print("I made it on episode: {} Reward: {}".format(episode,reward))
            q_table[get_discrete_state(state)][action] = 0
        state = next_state
```

Code from [Github Repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/1_q_learning_python_mountain_car.ipynb) with MIT lisence

### 5 查看训练效果

我们训练了 10000 次，将每500次的平均奖励，最大奖励，最小奖励结果画出来如下：

![image-20190705170404949](http://ww2.sinaimg.cn/large/006tNc79gy1g4persirrzj30ps0ggace.jpg)

可见，从大慨3000个回合的时候，智能体开始学会如何爬上山顶。

当然最直观的查看训练效果都方法即将动画render 出来，根据Q表来Render 动画的代码如下：

```python
done = False
state = env.reset()
while not done:
    action = np.argmax(q_table[get_discrete_state(state)])
    next_state, _, done, _ = env.step(action)
    state = next_state
    env.render()

env.close()
```

动画如下：

![MountainCar-v0](http://ww2.sinaimg.cn/large/006tNc79gy1g4pexb3vcag30xc0m8437.gif)

### 6. 总结

Q - learning 的关键在于如何建立Q-表，特别是处理环境状态为连续的情况，当然我们还会遇到行动空间同样为连续的情况，这种情况该如何处理呢？我们将在后面的文章介绍。

------

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

同步到我的简书 https://www.jianshu.com/u/bd506afc6fc1