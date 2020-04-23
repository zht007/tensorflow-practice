![Stormtrooper minifigure walking on the sand](https://ws1.sinaimg.cn/large/006tNc79gy1g2m2j7myavj30rs0ikq4l.jpg)

*image source from [unsplash.com](https://images.unsplash.com/photo-1472457897821-70d3819a0e24?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1049&q=80) by Daniel Cheung*

之前我们介绍了[Q-learning](https://steemit.com/ai/@hongtao/ai-q-learning)和S[ASAR算法](https://steemit.com/ai/@hongtao/ai-sarsa)的理论，这篇文章就理论结合实际用Q-learning 和SASAR算法指导智能体，完成悬崖探宝任务。

同样的，为了方便与读者交流，所有的代码都放在了这里：

https://github.com/zht007/tensorflow-practice

### 1. 环境简介

智能体在下图4 *12的格子世界中活动，"x"代表起点和智能体当前的位置，"T"代表终点，"C"代表悬崖，"o"代表其他位置。

```
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
x  C  C  C  C  C  C  C  C  C  C  T
```

> 状态(States)：当前位置
>
> 奖励(Rewards)：终点为1，悬崖为-100，其他地方为1
>
> 行动(Action)：上下左右，四个方向

### 2. SARSA算法

SARSA全称State–Action–Reward–State–Action，是on-policy的算法，即**只有一个策略**指挥行动并同时被更新。顾名思义，该算法需要5个数据，**当前的 state, reward，action**和**下一步state和action**。两步action和state均由epsilon greedy策略指导。

#### 2.1 定义epsilon greedy的策略

为了保证On-Policy的算法能访问到所有的状态，SARSA所有的行动策略必须是epsilon greedy的，这里定义的epsilon greedy策略与前文中是一样的。

```python
def make_epsilon_greedy_policy(Q, epsilon, nA):
    def policy_fn(observation):
        A = np.ones(nA, dtype=float) * epsilon / nA
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb) with MIT license*

#### 2.2 定义SARSA算法 

首先，根据**当前策略**迈出第一步，获得**当前的S和A**。

```python
policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
for i_episode in range(num_episodes):
        # First action
        state = env.reset()
        action_probs = policy(state)
        action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb) with MIT license*

然后，进入循环，直到游戏结束(if done: break)。该循环是为了获得**当前的R**和**下一步的S‘，和A‘**。并带入公式更新Q(S ,A)，由于策略是通过Q(S, A)生成的，所以更新Q(S,A)的同时，策略也更新了。
$$
Q(S, A) \leftarrow Q(S, A)+\alpha\left[R+\gamma Q\left(S^{\prime}, A^{\prime}\right)-Q(S, A)\right]
$$

```python
				while True:
            next_state, reward, done, _ = env.step(action)
            
            next_action_probs = policy(next_state)         
            next_action = np.random.choice(np.arange(len(action_probs)), 		 p=next_action_probs)
            
            Q[state][action] += alpha * (reward + discount_factor * Q[next_state][next_action] - Q[state][action])
            
            if done:
                break
            state = next_state
            action = next_action 
   return Q
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/SARSA%20Solution.ipynb) with MIT license*

将当前S和A替换成下一步S‘和A‘，直到游戏结束，最终得到优化后的Q表。

### 3. Q-Learning算法

Q-learning 与 SASAR有非常多的相似之处，但是本质上，Q-learning是Off-Policy的算法。也就是说Q-learning有两套Policy，Behavior Policy 和 Target Policy, 一个用于探索另一个用于优化。

与SARSA一样Q-Learning 也需要定义相同的epsilon greedy的策略，这里略过，我们看看算法本身的代码。

```python
policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    
    for i_episode in range(num_episodes):
        state = env.reset()
        
        while True:
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            best_action = np.argmax(Q[state])
            
            Q[state][action] += alpha * (reward + discount_factor * Q[next_state][best_action] - Q[state][action])
                  
            if done:
                break
            state = next_state
    
    return Q
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/TD/Q-Learning%20Solution.ipynb) with MIT license*

该算法与SARSA的区别是，Q-Learning在行动的时候**采用epsilon greedy的策略(Behavior Policy)**，但是在更新 Target Policy 的Q(S,A)时候，采用的是greedy的策略，即下一步的最大回报(best_action = np.argmax(Q[state]))

### 4. 总结

上文介绍的SASAR和Q-learning都属于单步Temporal Difference (时间差分TD(0))算法，其通用的更新公式为
$$
Q[s, a]+=\text { learning_rate } *\left(\text {td_target}-Q[s, a]\right)
$$
其中 td_target - Q[s,a] 部分又叫做 TD Error.

**SARSA算法:** 
$$
\text { td_target }=R[t+1] + \text {discout_factor}*Q[s',a']
$$
**Q-learning:**
$$
\text { td_target }=R[t+1] + \text {discout_factor}*max(Q[s'])
$$


关于两个算法的对比，我们可以看看两者最终的行动轨迹，首先定义渲染函数

```python
def render_evn(Q):
    state = env.reset()
    while True:
        next_state, reward, done, _  = env.step(np.argmax(Q[state]))
        env.render()
        if done:
            break
        state = next_state
```

SARSA算法

```python
Q1, stats1 = sarsa(env, 500)
render_evn(Q1)

----output---
x  x  x  x  x  x  x  x  x  x  x  x
x  o  o  o  o  o  o  o  o  o  o  x
x  o  o  o  o  o  o  o  o  o  o  x
x  C  C  C  C  C  C  C  C  C  C  T

```

Q-learing

```python
Q1, stats1 = sarsa(env, 500)
render_evn(Q1)

----output---
o  o  o  o  o  o  o  o  o  o  o  o
o  o  o  o  o  o  o  o  o  o  o  o
x  x  x  x  x  x  x  x  x  x  x  x
x  C  C  C  C  C  C  C  C  C  C  T
```

可以看出Q-learning得到的policy是沿着悬崖的最短(最佳)路径，获得的奖励最多，然而这样做却十分危险，因为在行动中由于采用的epsilon greedy的策略，有一定的几率掉进悬崖。SARSA算法由于是On-policy的在更新的时候意识到了掉进悬崖的危险，所以它选择了一条更加安全的路径，即多走两步，绕开悬崖。

---

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