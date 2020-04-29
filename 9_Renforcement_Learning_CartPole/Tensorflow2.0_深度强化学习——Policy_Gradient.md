![blue and brown wooden boat on water during daytime](https://tva1.sinaimg.cn/large/007S8ZIlgy1geaurrqwqnj30rs0ijaec.jpg)*image from unsplash.com by [@jodaarba](https://unsplash.com/@jodaarba)*

在之前的文章中我们系统地介绍了强化学习，以及与神经网络相结合的深度强化学习。期间由于 Tensorflow 2.0 尚未正式发布，大多数代码均使用 Tensorflow 1.x 或者 Keras 实现的，今后我们逐渐会用 Tensorflow 2.x 或者 PyTorch 更新代码，同时借机复习相关知识。

这篇文章我们还是借助 Open AI 的 CarPole 游戏，使用 Tensorflow 2.x 实现 Policy Gradient 算法完成游戏。

### 1. Policy Gradient 算法回顾

由于该部分已经在[前文](https://www.jianshu.com/p/f5d322a542ba)中做过详细的介绍，这里就不重复了。但是为了加深读者对 Policy Gradient 更深入的理论认识，这里截取了李宏毅老师的 PPT 并通过对这张PPT的解读加深对 Policy Gradient 算法的理解：

> * Policy 在这里既为θ，Rθ 是在该 Policy 下能够获得的奖励的期望，为了最大化Rθ，我们需要求Rθ的梯度，并通过**梯度上升**的方法优化θ的参数。
> * R(τ) 是游戏规则决定的，是不可微分的，而pθ(τ) 包含了在该Policy下行动的概率分布，是可以微分可以被优化的，所以求Rθ梯度的问题就转化成了求pθ(τ)梯度的问题。
> * 通过这个公式 $\nabla f(x)=f(x) \nabla \log f(x)$ 又可以将式子转换成 $\sum_{\tau} R(\tau) p_{\theta}(\tau) \nabla \log p_{\theta}(\tau)$
> * 其中$\sum_{\tau}p_{\theta}(\tau)$表示成期望的形式，而期望是可以通过采样来估算的，这里就将穷举的问题转化成了采样的问题。
> * $p_{\theta}\left(\tau^{n}\right)$由Policy决定的部分是$p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)$ （另外一个部分是状态转移概率由环境决定的 参考之前的对[MDP的介绍](https://www.jianshu.com/p/e63c75290d84)的文章)。
> * 所以最后 我们就将对 Rθ 求导的问题，转化成了通过采样对$p_{\theta}\left(a_{t}^{n} | s_{t}^{n}\right)$ 求导的问题。

![image-20200427130435916](https://tva1.sinaimg.cn/large/007S8ZIlgy1ge8kw7o70vj31450u0dmz.jpg)

李宏毅老师的这页PPT清楚地解释了什么是 Policy 需要优化的目标函数，怎么样从一个无法操作的穷举问题转化成一个可以计算的的采样问题。

###  2. Tensorflow 2.x 实战

代码部分大多数地方与 Tensorflow 1.x 相同仅仅在 Agent 对象部分有所改变。

在 Tensorflow 1.x 中神经网络是在 Agent 对象里面，在 Tensorflow 2.x 一般是单独定义 Network 对象,  并在 Agent 内部实例化，这里我们搭建了一个两层全连接的神经网络。

```python
class Network(keras.Model):
  def __init__(self):
    super().__init__()

    self.model = keras.Sequential(
        [layers.Dense(10, 'relu'),
        layers.Dense(ACTION_SPACE_SIZE)]
    )

  def call(self, x):
    out = self.model(x)

    return out
```

然后，在 Agent Class 中实例化这个神经网络，在训练部分采用 Tensorflow 2.x 的方法既`tf.GradientTape()`包裹需要求梯度的部分。

```python
class PGAgent:

  def __init__(self):
    self.network = Network()
    self.network.build(input_shape = (None,OBSERVATION_SPACE_SIZE))

  def choose_action(self, states):
    states = states.reshape(-1, OBSERVATION_SPACE_SIZE)

    action_logits = self.network(states)
    actions_prob = tf.nn.softmax(action_logits)
    action = np.random.choice(len(actions_prob.numpy()[0]),p=actions_prob.numpy()[0])
    return action
  def train(self, states, actions, rewards):
    
    discounted_episode_rewards = self.discount_rewards(rewards)
    optimizer = keras.optimizers.Adam(0.001)
    
    with tf.GradientTape() as tape:
      action_logits = self.network(states)
      cross_entropy = tf.losses.sparse_categorical_crossentropy(y_true=actions, 
                                                                   y_pred=action_logits,
                                                                   from_logits=True )
      loss = tf.reduce_mean(cross_entropy * discounted_episode_rewards)

    grads = tape.gradient(loss, self.network.variables)
    optimizer.apply_gradients(zip(grads, self.network.variables))


  def discount_rewards(self,rewards):
        # discount episode rewards
        discounted_ep_rs = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * GAMMA + rewards[t]
            discounted_ep_rs[t] = running_add

        # normalize episode rewards
        discounted_ep_rs -= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs
```

------

相关文章

[强化学习——MC(蒙特卡洛)玩21点扑克游戏](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fcn-stem%2F%40hongtao%2Fmc-21)
[强化学习实战——动态规划(DP)求最优MDP](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fcn-stem%2F%40hongtao%2Fdp-mdp)
[强化学习——强化学习的算法分类](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2F7atbof)
[强化学习——重拾强化学习的核心概念](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2F2bqdkd)
[AI学习笔记——Sarsa算法](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2Fai-sarsa)
[AI学习笔记——Q Learning](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2Fai-q-learning)
[AI学习笔记——动态规划(Dynamic Programming)解决MDP(1)](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2Fai-dynamic-programming-mdp-1)
[AI学习笔记——动态规划(Dynamic Programming)解决MDP(2)](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2Fai-dynamic-programming-mdp-2)
[AI学习笔记——MDP(Markov Decision Processes马可夫决策过程)简介](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2Fai-mdp-markov-decision-processes)
[AI学习笔记——求解最优MDP](https://links.jianshu.com/go?to=https%3A%2F%2Fsteemit.com%2Fai%2F%40hongtao%2Fai-mdp)

