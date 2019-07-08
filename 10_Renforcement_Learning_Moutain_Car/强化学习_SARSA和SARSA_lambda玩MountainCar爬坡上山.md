![landscape photography of road between mountains](http://ww1.sinaimg.cn/large/006tNc79gy1g4so4ks1vqj30rs0eb770.jpg)

Image from [unsplash.com](https://unsplash.com/photos/Xj4thPRvSSs) by Jonatan Pie

[上一篇文章](https://steemit.com/cn-stem/@hongtao/q-learning-mountaincar)我们介绍了用 Q-learning 的算法完成了小车爬坡上山的游戏，这篇文章我们来讲讲如何用 SARSA 算法完成同样挑战。

### 1. Q-Learning 和 SARSA 异同

Q - Learning 和 SARSA 有很多相似之处，他们均属于单步Temporal Difference (时间差分TD(0)算法，其通用的更新公式为
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
不同之处在于，SARSA 是On-policy 的算法，即**只有一个策略**指挥行动并同时被更新。Q - Learning 是 Off-Policy 的算法， 探索的时候采用max 的策略来更新Policy(Q表)，但行动的时候未必会走 max 奖励的那条路 (epsilon greedy 策略)。

由于两者的相似度很高，在代码中仅需要小小的改动即可实现 SARSA 算法。

同样的，为了方便与读者交流，所有的代码都放在了这里：

https://github.com/zht007/tensorflow-practice

### 3. SARSA 算法实现

Q表的建立，帮助函数等都与Q-Learning 一模一样，这里就不赘述了。仅仅在训练阶段有所不同。

> 首先，在初始化环境(state = env.reset()) 的同时，需要初始化 action (action = take_epilon_greedy_action(state, epsilon))
>
> 其次，获取 next_action (next_action = take_epilon_greedy_action(next_state, epsilon))
>
> 然后，依据前面的公式， td_target 中的 max 部分由下一个状态的 q[next_sate, next_action] 替换。
>
> 最后，action = next_action 将本循环中的 next_action 带入到下一个循环中



完整核心代码如下

```python
for episode in range(EPISODES):

    state = env.reset()
    action = take_epilon_greedy_action(state, epsilon)
    
    done = False
    while not done:

        next_state, reward, done, _ = env.step(action)

        ep_reward += reward
        
        next_action = take_epilon_greedy_action(next_state, epsilon)

        if not done:

            td_target = reward + DISCOUNT * q_table[get_discrete_state(next_state)][next_action]

            q_table[get_discrete_state(state)][action] += LEARNING_RATE * (td_target - q_table[get_discrete_state(state)][action])

        elif next_state[0] >= 0.5:
            q_table[get_discrete_state(state)][action] = 0


        state = next_state
        action = next_action

```

*Code from [Github Repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/2_SARSA_python_mountain_car.ipynb) with MIT license*

### 4. SARSA lambda 算法

Q learning 和 SARSA 都是**单步更新(TD(0))**的算法。单步跟新的算法缺点就是在没有到达目标之前的那些『原地打转』的行动也被记录在案，每走一步，脚下的Q-表也被更新了，虽然这些行动是毫无意义的。

SARSA Lambda(λ)， 即引入 **λ** 这个衰减系数，来解决这个问题。与γ用来衰减未来预期Q的值一样，λ是当智能体获得到达目标之后，在更新Q表的时候，给机器人一个回头看之前走过的路程的机会。相当于，机器人每走一步就会在地上插一杆旗子，然后机器人每走一步旗子就会变小一点。

于是我们需要另一个表，来记录每一个 state 的查旗子的状态(大小)，这就需要一个与Q-表相同的**eligibility trace 表** (E表)。

```python
LAMBDA = 0.95

e_trace = np.zeros((DISCRETE_OS_SIZE + [env.action_space.n]))
```

与Q learning 和 SARSA 智能体每走一步仅更新脚下的Q表 (当前状态的Q(S, A))不同，SARSA lambda 每走一步，整个Q 表(和 E 表)都会被更新。完整算法如下：

![image-20190708124040344](http://ww1.sinaimg.cn/large/006tNc79gy1g4so0nco2qj30i20akwiy.jpg)

*Image from [[1]](http://incompleteideas.net/book/RLbook2018.pdf) by Sutton R.S.*

> 首先，每一个回合均需要将 eligibility trace 表初始化为0。
>
> 其次，用 delta 来表示下一个状态的Q值和当前状态Q值的差值，并在当前状态"插旗"(E(S,A) += 1)。
>
> 最后，更新Q表和E表。

完整核心代码如下：

```python
for episode in range(EPISODES):

    state = env.reset()
    action = take_epilon_greedy_action(state, epsilon)
    
#   reset e_trace IMPORTANT
    e_trace = np.zeros(DISCRETE_OS_SIZE + [env.action_space.n])
    
    done = False
    while not done:

        next_state, reward, done, _ = env.step(action)

        ep_reward += reward
        
        next_action = take_epilon_greedy_action(next_state, epsilon)

        if not done:
            
            delta = reward + DISCOUNT * q_table[get_discrete_state(next_state)][next_action] - q_table[get_discrete_state(state)][action]

            e_trace[get_discrete_state(state)][action] += 1
            
            q_table += LEARNING_RATE * delta * e_trace
            
            e_trace = DISCOUNT * LAMBDA * e_trace
            
        elif next_state[0] >= 0.5:
            q_table[get_discrete_state(state)][action] = 0


        state = next_state
        action = next_action

    # Decaying is being done every episode if episode number is within decaying range
    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
```

*Code from [Github Repo](https://github.com/zht007/tensorflow-practice/blob/master/10_Renforcement_Learning_Moutain_Car/3_SARSA_lambda_python_mountain_car.ipynb) with MIT license*

### 4. Q-Learning, SARSA, SARSA lambda 对比

为了方便对比，我们将初始Q表的值均设置为0，表的长度均为20，跑10,000个回合, 每200个回合计算一下最小，平均和最大奖励。结果如下图所示：

![episodes vs rewards](http://ww1.sinaimg.cn/large/006tNc79gy1g4sng88zaxj30uq0ljtbq.jpg)

可以看出，三个算法均在4000 个回合开始 Converge。 但是由于是 Off - Policy 的算法，Q-learning 在最小奖励的表现不如SARSA算法，说明Q-learning 的智能体，更加大胆，勇于探索最大化奖励，SARSA 算法表现得更加谨慎，时刻遵守Policy的指导，平均奖励优于Q-learning.

-----

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