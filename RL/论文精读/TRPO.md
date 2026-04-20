# TRPO-Trust Region Policy Optimization

## 前景提要
我们之前刚刚穿越了强化学习的数学原理，从贝尔曼方程-贝尔曼最优方程-值迭代与策略迭代-MC-随机近似-TD-值函数近似-策略梯度-AC算法

在最后，我们的AC算法以及A2C算法都是属于策略梯度的算法，这种方式通过将策略$ \pi$参数化,并且通过重要性采样变成off-policy的A2C，引入优势函数



[(59 封私信 / 81 条消息) 深度强化学习（三）：TRPO（Trust Region Policy Optimization ，信赖域策略优化） - 知乎](https://zhuanlan.zhihu.com/p/605886935)

TRPO欢迎去看这个，因为这个写的已经非常的全了，

PPO是John Schulman在2015年提出的，是Schulam的博士论文



## 问题提出

有一个问题是这样的，我们在训练的时候会发现，我们的算法很难保证**单调收敛**，此外，我们的学习率$\alpha$也很难去做

首先说一下什么叫做*单调收敛*

回顾一下我们的算法
$$
\theta_{new} = \theta_{old} + \alpha \nabla_{\theta} J(\pi_{\theta})
$$
这种其实是在数学上，是对$\theta$的一阶进行展开来去求的（看看是不是特别想泰勒公式在一阶的展开，也就是所谓的梯度下降），如果我们的学习率过大的话，会突然间跳过这里

其次，参数空间是一个高维度的空间，但是我们的学习率是一个欧几里得的固定的学习率，这样做会导致更新的时候策略产生剧烈的震荡

最后其实就是这个东西没有什么下界的约束



上面说的这三个原因总结一下就是，**策略更新的时候会剧烈震动**，那么我们想要的是什么？是让这个策略**越来越好**



什么叫做越来越好？就是说，我每更新一步，我的策略所带来的回报都是比上一步正的，即我需要保证**策略单调收敛**

这样的好处不仅仅是策略可以稳定，由于我们是on policy的，所以说如果策略越来越好的话，那么我们的数据也是越来越好的，相反，之前的不稳定就是会产生很坏的数据，影响训练



那么要怎么做才能使我们的策略更新的时候是单调的呢

一个直接的想法是，我们看一下当前的能不能拆成 之前的+某个东西，我们约束某个东西是正的就行了，这样就可以保证一直是增的就行

## 策略性能等式

 Sham Kakade 和 John Langford 在 2002 提出这个等式，其实就是我们之前说的，当前的能不能拆成 之前的+某个东西



首先我们需要定义策略的期望回报，即
$$
\eta(\pi) = E_{s_0,a_0...\sim\pi}[\sum^{\infty}_{t = 0}\gamma^tr(s_t,a_t)]=E_{s_0}[V^\pi(s_0)]
$$
这里就是对一个策略的评价，即遵循策略，从s0出发得到的期望回报

我们之前学过优势函数和Q，
$$
Q^\pi (s_t,s_t) = r(s_t,a_t)+\gamma E_{s_{t+1}}[V^\pi(s_{t+1})]
$$
优势函数其实就是Q-V，即
$$
A^\pi(s_t,a_t) =Q^\pi (s_t,s_t) -V^{\pi}(s_t)
$$
其实就是
$$
A^\pi(s_t,a_t) = r(s_t,a_t)+\gamma E_{s_{t+1}}[V^\pi(s_{t+1})]-V^{\pi}(s_t)
$$
考察新策略 $\tilde{\pi}$ 在其采样的轨迹上，旧策略 $ \pi$ 的优势函数累积期望：
$$
\mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t) \right]
$$
我们把刚才的A的带入后就是
$$
\begin{align}
\mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t) \right]  

&=\mathbb{E}_{\tau \sim \tilde{\pi}} \left [\sum_{t=0}^{\infty} \gamma^t(r(s_t,a_t)+\gamma V^\pi(s_{t+1})-V^{\pi}(s_t))\right]\\


  &=\mathbb{E}_{\tau \sim \tilde{\pi}} \left [ \sum_{t=0}^{\infty}\gamma^t r(s_t,a_t) +  \sum_{t=0}^{\infty} \gamma^{t+1}V^\pi(s_{t+1})- \sum_{t=0}^{\infty}\gamma^t    V^{\pi}(s_t)) \right]\\
  
  &=\mathbb{E}_{\tau \sim \tilde{\pi}} \left [ \sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)  -V^\pi(s_0)\right]\\
  &=-\mathbb{E}_{\tau \sim \tilde{\pi}}V^\pi(s_0) + \mathbb{E}_{\tau \sim \tilde{\pi}}\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\\
  &=-\eta(\pi)+\eta(\tilde{\pi}) 
  
\end{align}
$$
1-2行就是带入进去

2-3行是因为，注意看后面的$\sum_{t=0}^{\infty} \gamma^{t+1}V^\pi(s_{t+1})- \sum_{t=0}^{\infty}\gamma^t    V^{\pi}(s_t))$,有没有发现，这里面有很多是可以消掉的，

左边是从$\gamma$的1次方，一直到无穷次方，右边是从0次方一直到无穷次方，那么两个相减，最后就只剩下0次方了，

4-5行是因为因为初始状态 $s_0$ 的分布完全由环境决定，与智能体采用何种策略（$ \pi$ 还是 $\tilde{\pi}$）无关，所以：

$$\mathbb{E}_{\tau \sim \tilde{\pi}} [-V^\pi(s_0)] = \mathbb{E}_{s_0} [-V^\pi(s_0)] = -\eta(\pi)$$

最后就是对右边的完整的折旧期望回报的和G_t，也就是V，

所以我们有
$$
\eta(\tilde{\pi}) = \eta(\pi) + \mathbb{E}_{\tau \sim \tilde{\pi}} \left[ \sum_{t=0}^{\infty} \gamma^t A^\pi(s_t, a_t) \right]
$$
这样，我们就把一个新策略衡量问一个就策略+一个期望的形式，TRPO要从这里开始出发，让后面这一坨大于等于0，从而得到目的



拆开来看，就是
$$
\eta(\tilde{\pi}) = \eta(\pi) + \sum_{s} \rho_{\tilde{\pi}}(s) \sum_{a} \tilde{\pi}(a|s) A^\pi(s, a)
$$




## 代理函数

其实我们是没有后面那项的前面的，即

图中第一段文本指出了直接优化这个等式的痛点：等式中包含了 **$\rho_{\tilde{\pi}}(s)$（新策略的状态分布）**。因为我们在更新参数之前，新策略 $\tilde{\pi}$ 还是未知的，所以我们无法知道它在环境中运行时会产生什么样的状态分布，这个等式也就无法求解。



所以我们需要近似，

**TRPO技巧一**

TRPO 强制忽略状态分布的变化，用**旧策略的状态分布 $\rho_{\pi}(s)$** 强行替换未知的 $\rho_{\tilde{\pi}}(s)$。



为什么，因为

文本中提到“当新旧参数很接近时，用旧的状态分布代替新的状态分布也是合理的”。在数学上，只要我们能够通过某种约束（后续引入的 KL 散度）保证新策略 $\tilde{\pi}$ 和旧策略 $ \pi$ 的差距非常小，那么 $\rho_{\tilde{\pi}}(s)$ 和 $\rho_{\pi}(s)$ 的差距也会是一个二阶小量（可以被忽略）。



即
$$
L_\pi(\tilde{\pi}) = \eta(\pi) + \sum_{s} \rho_\pi(s) \sum_{a} \tilde{\pi}(a|s) A^\pi(s, a)
$$
所以我们优化旧策略就行了

## 衡量策略之间的差异

KL散度

TV散度

Pinsker不等式



这一部分看论文



不等式右边就是新策略的性能下界



![image-20260419153520868](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419153520868.png)

## 引入置信域

注意到这里有一个C

C通常是很大的，但是呢，

![image-20260419153724842](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419153724842.png)

这样就是一个带约束条件的置信域的问题



我们把最大KL散度更改为平均KL散度，因为如果是最大KL散度，各个条件下的KL散度都是小于的，条件就很多，难以实现，



平均KL散度就是旧策略访问到的状态的平均KL散度



## 重要性采样

这个会了

因为之前的L是从新策略采样的，但是只能从旧策略采样，所以需要引入重要性采样，加一个权重就行 



## 近似

目标L在$\theta$等于old的一节泰勒

约束条件做了二阶泰勒

![image-20260419154301487](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419154301487.png)

后面就看不懂了，就是怎么去求这个hessian矩阵，然后怎么去解这个问题

最后求解了一个带不等式约束的最大化问题，用共轭梯度去解等等



