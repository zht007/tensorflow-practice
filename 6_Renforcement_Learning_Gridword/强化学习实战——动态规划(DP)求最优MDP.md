![top view of ground during daytime](https://ws3.sinaimg.cn/large/006tNc79gy1g29atytz5uj30rs0ign26.jpg)

 *image source from [unsplash](https://images.unsplash.com/photo-1494373306878-b0f432fe7288?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&auto=format&fit=crop&w=1353&q=80) by Stijin te Strake*

[之前的文章](https://steemit.com/ai/@hongtao/ai-dynamic-programming-mdp-2)介绍了用动态规划(DP: Dynamic Programming)求解最优MDP的理论。DP求解最优MPD有两个方法，一是**策略迭代(Policy Iteration)**，另一个就是**值迭代(Value Iteration)**。本篇文章就用Python编程实践这个理论。

同样的，为了方便与读者交流，所有的代码都放在了这里：

<https://github.com/zht007/tensorflow-practice>



### 1. 策略迭代(Policy Iteration)

策略迭代求解MDP需要分成两步，第一步是策略评估(Policy Evaluation)，即用Bellman等式评估当前策略下的MDP值函数，直到值函数稳定收敛；第二步是根据这个收敛的值函数迭代策略，最终获得最佳MDP。

这里还是以[前文](https://steemit.com/ai/@hongtao/ai-dynamic-programming-mdp-2)的格子世界(Grid World)为例：

```
    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T
```

> * 机器人在4x4的格子世界中活动，**目的地**是左上角或者右下角。
>
> * x为机器人现在所在地**位置(状态S)**。
>
> * 机器人的**行动A**有四个方向 (UP=0, RIGHT=1, DOWN=2, LEFT=3)。
>
> * 除**目的地**外，其他地方的奖励(reward)都为"-1"

#### 1.1 策略评估(Policy Evaluation)

首先"/ilb/envs"目录下已经将Grid World环境搭建好了，其中

> **S** 是一个16位的向量，代表状态0到15。env.nS 可以得到所有的状态数(16)。
>
> **A**是一个4位的向量，代表行动方向0到3。env.nA 可以得到行动数量(4)。
>
> **policy** 是一个16 x 4 的矩阵，代表每个状态s下，改行动a的概率(action probability)
>
> 调用**env.P[s] [a]**可以得到一个list 包括(prob, next_state, reward, done)，注意这里的prob是**状态转移**概率.

策略评估的函数如下

```python
def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    while True:
        delta = 0
        # For each state, perform a "full backup"
        for s in range(env.nS):
            v = 0
            # Look at the possible next actions
            for a, action_prob in enumerate(policy[s]):
                # For each action, look at the possible next states...
                for  prob, next_state, reward, done in env.P[s][a]:
                    # Calculate the expected value. Ref: Sutton book eq. 4.6.
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # How much our value function changed (across any states)
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        # Stop evaluating once our value function change is below a threshold
        if delta < theta:
            break
    return np.array(V)
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb) with MIT license*

这里涉及到多个循环，最外层"while"循环是迭代循环；第一个"for"循环是遍历所有状态；第二个'for‘循环是遍历该s的policy，得到a 和 action probiliby；最里层的'for'循环最为重要，调用的是**Bellman Equation**。

```python
 v += action_prob * prob * (reward + discount_factor * V[next_state])
```

我们将随机策略带入函数中，最终获得收敛的值函数。

```python
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)


--V output--
Grid Value Function:
[[  0.         -13.99993529 -19.99990698 -21.99989761]
 [-13.99993529 -17.9999206  -19.99991379 -19.99991477]
 [-19.99990698 -19.99991379 -17.99992725 -13.99994569]
 [-21.99989761 -19.99991477 -13.99994569   0.        ]]
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb) with MIT license*

#### 1.2 策略改进(Policy Improvement)

根据策略评估得到的V函数，通过Greedy的方法选择下一步值函数最大的行动，用这种方法迭代改进策略，就能得到最佳策略。

这里需要创建帮助函数，one step lookahead，看看在该状态下不同的行动获得的状态函数是多少，同样使用了**Bellman Equation**

```python
    def one_step_lookahead(state, V):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb) with MIT license*

接着，将策略评估得到的值函数带入"one_step_lookahead"得到不同行动下的状态函数。选择"获利"最大的行动，将这个行动的action probablity设为"1"，其他action设为"0" , 这样即完成了策略改进。最后通过迭代可获得最佳策略和最佳值函数。

```python
policy = np.ones([env.nS, env.nA]) / env.nA
    
    while True:
        # Evaluate the current policy
        V = policy_eval_fn(policy, env, discount_factor)
        
        # Will be set to false if we make any changes to the policy
        policy_stable = True
        
        # For each state...
        for s in range(env.nS):
            # The best action we would take under the currect policy
            chosen_a = np.argmax(policy[s])
            
            # Find the best action by one-step lookahead
            # Ties are resolved arbitarily
            action_values = one_step_lookahead(s, V)
            best_a = np.argmax(action_values)
            
            # Greedily update the policy
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        # If the policy is stable we've found an optimal policy. Return it
        if policy_stable:
            return policy, V
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb) with MIT license*

运行policy_improvement 函数得到收敛的最佳策略和最佳值函数。

```python
policy, v = policy_improvement(env)

---output--
Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):
[[0 3 3 2]
 [0 0 0 2]
 [0 0 1 2]
 [0 1 1 0]]

Reshaped Grid Value Function:
[[ 0. -1. -2. -3.]
 [-1. -2. -3. -2.]
 [-2. -3. -2. -1.]
 [-3. -2. -1.  0.]]
```

### 2 值迭代(Value Iteration)

值迭代不必对策略进行评估，直接将"one_step_lookahead"评估出的**最大值**带入，同时将能得到最大值的的**行动**作为新的策略，最终可以同时得到最佳值函数和最佳策略。

#### 2.1 值更新

取出"one_step_lookahead"中最大的值更新值函数的value.

```python
    V = np.zeros(env.nS)
    while True:
        # Stopping condition
        delta = 0
        # Update each state...
        for s in range(env.nS):
            # Do a one-step lookahead to find the best action
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Update the value function. Ref: Sutton book eq. 4.10. 
            V[s] = best_action_value        
        # Check if we can stop 
        if delta < theta:
            break
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb) with MIT license*

#### 2.2 策略更新

"one_step_lookahead"中最大值对应的行动a，即为新策略的a，action probability 设为"1"，其他action 设为'0'

```python
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
```

*该部分代码参考[github](https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Iteration%20Solution.ipynb) with MIT license*

运行value_iteration 函数得到与policy iteration相同的最佳策略和最佳值函数。

```python
policy, v = value_iteration(env)

----output------
Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):
[[0 3 3 2]
 [0 0 0 2]
 [0 0 1 2]
 [0 1 1 0]]
 
 Reshaped Grid Value Function:
[[ 0. -1. -2. -3.]
 [-1. -2. -3. -2.]
 [-2. -3. -2. -1.]
 [-3. -2. -1.  0.]]

```



-------

参考资料

[1] [Reinforcement Learning: An Introduction (2nd Edition)](http://incompleteideas.net/book/RLbook2018.pdf)

[2] [David Silver's Reinforcement Learning Course (UCL, 2015)](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)

[3] [Github repo:  Reinforcement Learning](https://github.com/dennybritz/reinforcement-learning) 

-----

相关文章

[强化学习——强化学习的算法分类](https://steemit.com/ai/@hongtao/7atbof)

[强化学习——重拾强化学习的核心概念](https://steemit.com/ai/@hongtao/2bqdkd)

[AI学习笔记——动态规划(Dynamic Programming)解决MDP(1)](https://steemit.com/ai/@hongtao/ai-dynamic-programming-mdp-1)

[AI学习笔记——动态规划(Dynamic Programming)解决MDP(2)](https://steemit.com/ai/@hongtao/ai-dynamic-programming-mdp-2)

[AI学习笔记——MDP(Markov Decision Processes马可夫决策过程)简介](https://steemit.com/ai/@hongtao/ai-mdp-markov-decision-processes)

[AI学习笔记——求解最优MDP](https://steemit.com/ai/@hongtao/ai-mdp)