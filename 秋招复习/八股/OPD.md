# OPD

## KL散度

kl散度分为forward和reverse kl

假设q是学生，是我们的要训练的分布，p是老师，是我们要拟合的分布
$$
D_{FKL} = D(p(x)||q(x)) = \sum_{x}p(x)\log(\frac{p(x)}{q(x)})
$$

$$
D_{rKL} = D(q(x)||p(x)) = \sum_{x}q(x)\log(\frac{q(x)}{p(x)})
$$

先看一下forward KL

展开后就是
$$
D_{FKL} = D(p(x)||q(x)) = \sum_{x}p(x)\log p(x) - \sum_{x}p(x)\log q(x)
$$
如果我们要对q求的话，其实就是一个比较典型的交叉熵，因为前面的p都是没有的
$$
\nabla D(p(x)||q(x)) = -\nabla \sum_{x}p(x)\log(q(x))  = E_{x～p(x)}\nabla\log q(x)
$$
这个跟我们的sft是一样的，如果是one-hot的，只有一个是正确答案，比如x取y的时候，那么这个交叉熵就直接是$ -\log q(y)$

此时x是在p(x)上的分布（因为前面可以写成是sum px 就是一个典型的期望

如果是reverse KL呢，那么其实就是
$$
D_{RKL} =\sum_x q(x) \log q(x) - \sum_x q(x)\log (p(x))
$$
即
$$
D_{RKL} =-H(q) - E_{x～q}  \log p(x)
$$
相当于，模型在q采样后，觉得自己在q上多合理，在p上又有多合理



这里给一个例子

![image-20260803201058448](https://cdn.jsdelivr.net/gh/ShiangHou/markdown_image/imgimage-20260803201058448.png)

很显然，我们注意到，q在概率很低的时候，forward的时候q是在分母的，所以此时会导致这个KL迅速上升，即此时我们是想拟合p所有的分布



反观reverse，第二个例子，如果P是很低的，但是Q很高，此时reverse KL的P是在分母，那么这个惩罚就很大了，即reverse避免了q在p概率很低的时候还输出，而forward是避免p很高的时候q很低



换句话说，forward要求模型尽可能覆盖自己所有的分布，即model seeking，即“向0惩罚”，而reverse要求拟合出最大的分布，



因此，Reverse KL更倾向于选择一个高概率的答案，在生成、RL等推理任务中，更尖锐



比如，对于数学推理这种任务来说，天然就是需要一个稳定的推理线路，reverse KL就比较适配

## 测度

所谓的测度，即从谁的分布里面去采样，

对于reverse KL，注意到
$$
D_{RKL} =\sum_x q(x)( \log q(x) - \log (p(x))) = E_{x～q(x)}( \log q(x) - \log (p(x)))
$$
由于q就是在前面，而且我们的on policy就是自己采样的q，所以天然适配，

结合之前的reverse 这种模式，加上符合on-policy的测度，这就是我们选择reverse的原因

那么我们看一下Forward KL，不难发现，
$$
D_{FKL} = D(p(x)||q(x)) = \sum_{x}p(x)(\log p(x) - \log q(x)) = E_{x～p(x)}\log p(x) - \log q(x)
$$
这个其实是x在p分布上的，即教师分布，但是我们是要求 on-policy的，所以这个x一定要来自q，不能来自p

有聪明的骚猪已经想到了，我们可以用重要性采样，这是之前我们在RL里面常用的

对于forward的KL散度，我们直接
$$
D(p(x)||q(x)) = \sum_{x}p(x)(\log p(x) - \log q(x))  = \sum_x \frac{q(x)}{q(x)} p(x)(\log p(x) - \log q(x)) = E_{x～q} \frac{p(x)}{q{x}}(\log p(x) - \log q(x))
$$
注意到
$$
E_{x～q} \frac{p(x)}{q{x}}(\log p(x) - \log q(x)) =  E_{x～q} \frac{p(x)}{q(x)}\log \frac{p(x)}{q(x)}
$$
这个其实就是$\omega \log \omega$的形式，这个是得到两个q和p的量就可以算，但是我们看一下二介距
$$
E_{x～q} (\frac{p(x)}{q(x)})^2 \log ^2 \frac{p(x)}{q(x)} =\sum_x q(x) \frac{p(x)^2}{q(x)^2}\log ^2 \frac{p(x)}{q(x)} = \sum_x  \frac{p(x)^2}{q(x)}\log ^2 \frac{p(x)}{q(x)}
$$
回顾一下forward的概念，这个是要求q全部覆盖掉p，因为对于p大的q小的情况下要着重处理

但是，我们采样是从q采样的啊，如果q很小，那么我们其实几乎采样不到，如果采样采到的话，看这个q在分母，也会爆炸，方差也很大

这个是forward kl的计算范式要求与on-policy采样冲突的地方

## Top-K

所谓的topk截断，无论是哪个方向的，都是
$$
D_{KL}(p_t\|q_t)\approx\sum_{v\in\mathrm{top}\text{-}k(p_t)}p_t(v)\log\frac{p_t(v)}{q_t(v)}
$$
如果我们是对teacher模型的top k截断的话，更适合forward KL，因为本来外层就是对p采样的，但是教师的topk截断对reverse不是很契合，因为reverse前面是q，是自己的，但是如果对学生截断反而合理了

这其实解释了,verl里面只有forward 的topk的loss是true。verl 的接口设计：`forward_kl_topk` 是唯一标了 `use_topk=True` 的 loss，必须拿到 `teacher_logprobs` + `teacher_ids` 两个 $(bsz, seqlen, 64)$ 的张量
$$
\widetilde\ell_t(\theta;s_t)
=\sum_{v\in K_t}p_t(v\mid s_t)\left[\log p_t(v\mid s_t)-\log q_\theta(v\mid s_t)\right]
$$
，teacher 已经显式传来了 $K_t $ 中每个 token 的概率和 ID，student 对这些 ID 做 gather：）；而 `k1/k2/k3/low_var_kl/abs/mse` 全部走 `use_estimator=True`，只要采样 token 上的一个标量。***\*接口的形状是被 KL 的方向逼出来的。\****



- verl 的 `forward_kl_topk + use_policy_gradient=False` 则是在学生前缀上直接反传 teacher-top-k loss：
    - $y_t \sim q_{\text{rollout}}(\cdot\mid s_t), \quad s_t=(x,y_{<t})$
    - 记 $\mathcal K_t=\text{top-}k\big(p_t(\cdot\mid y_{<t})\big)$ 是教师在位置 $t$ 的 top-k token 集合（代码里的 teacher_topk_ids，$k$ 默认 64）。逐 token 损失是：


$$
\ell_t(\theta)=\sum_{v\in\mathcal{K}_t}p_t(v)\left[\log p_t(v)-\log q_\theta\!\left(v\mid y_{<t}\right)\right]
$$

- forward_kl_topk 应配 use_policy_gradient=False；若设为 true (`advantages = -distillation_losses.detach()`)，verl 自己会警告 top-k 的非采样 token 信号基本没被利用：
    - 在同一个生成位置 $t$，teacher 给出的 top-k 候选中，没有被 student rollout 实际选为 $y_t$ 的那些 token。

```python
if self.use_policy_gradient and self.loss_mode == "forward_kl_topk":
    print("WARNING: forward_kl_topk is most effective as a supervised distillation loss (use_policy_gradient=False). With policy gradient, the update uses only the sampled token's logprob ∇logπ(a), so the top-k distributional signal (how non-sampled logits should move) is largely unused.")
```

## reuse RL infra

>  reverse KL最小化等价于RL最大化的问题，和policy gradient是一样的，这也是为什么OPD为什么可以无缝衔接RL流程和infra的原因

对于一个RL来说，梯度是
$$
\nabla_\theta \mathcal{L}_{RL} = -\mathbb{E}_{y \sim \pi_{\theta}}[\nabla_{\theta}\log \pi_{\theta}(y)A(y)]
$$


对于reverse KL的loss是
$$
\mathcal{L}_{OPD}(\theta) = D_{KL}(\pi_{\theta} || \pi_{E}) = \sum_{y} \pi_{\theta}(y) \log \frac{\pi_{\theta}(y)}{\pi_{E}(y)}
$$
推一下，
$$
\nabla_{\theta} D_{KL}(\pi_{\theta} || \pi_{E}) = \sum_{y} \left( \nabla_{\theta}\pi_{\theta}(y) \cdot \log \frac{\pi_{\theta}(y)}{\pi_{E}(y)} + \pi_{\theta}(y) \cdot \nabla_{\theta} \left[ \log \pi_{\theta}(y) - \log \pi_{E}(y) \right] \right)
$$


注意到右侧部分 $\pi_{\theta}(y) \cdot \nabla_{\theta}\log\pi_{\theta}(y) \rightarrow \sum_{y} \nabla_{\theta}\pi_{\theta}(y) = \nabla_{\theta} \left( \sum_{y} \pi_{\theta}(y) \right)=0$ 即对一个E和值常数求导就是0

然后右边后面那个对常数求的也是0

继续，用我们的经典替换，A导等于A*导logA
$$
\nabla_{\theta} D_{KL} = \sum_{y} \nabla_{\theta}\pi_{\theta}(y) \cdot \log \frac{\pi_{\theta}(y)}{\pi_{E}(y)}=\sum_{y} \pi_{\theta}(y) \nabla_{\theta}\log\pi_{\theta}(y) \cdot \log \frac{\pi_{\theta}(y)}{\pi_{E}(y)}= - \mathbb{E}_{y \sim \pi_{\theta}} \left[ \nabla_{\theta}\log\pi_{\theta}(y) \cdot \log \frac{\pi_{E}(y)}{\pi_{\theta}(y)} \right]
$$
所以我们直接定义：$A(y) = \log \frac{\pi_{E}(y)}{\pi_{\theta}(y)}$

​    \- $A_t = \text{sg}\left[\log\frac{\pi_{E_i}(y_t|x, y_{<t})}{\pi_\theta(y_t|x, y_{\le t})}\right]$
$$
\log\frac{\pi_T(a_t|s_t)}{\pi_\theta(a_t|s_t)}=\log\pi_T(a_t|s_t)-\log\pi_\theta(a_t|s_t).
$$


这个就是reward/advantage，也就是我们把loss计作相反数
$$
d_t=\log\pi_\theta(a_t|s_t)-\log\pi_T(a_t|s_t).
$$


如果 student 给某 token 很高概率，但 teacher 给低概率，那么：$\log\pi_\theta-\log\pi_T > 0$ loss 高，应该压低它。

如果 teacher 比 student 更认可这个 sampled token，那么：$\log\pi_\theta-\log\pi_T < 0$ 它变成正向学习信号。

## verl

- verl 的主线设计是 HybridFlow:
    - 单进程 controller 表达 RL 算法控制流,
    - 多进程 worker 承担 actor、 rollout、critic、reward model、teacher model 等重计算。
    - OPD 没有另起一个完全独立的 trainer,而是接入已有 PPO 主干

-------

> PPO、GRPO、OPD

PPO、GRPO、OPD 共享同一条 trainer 骨架: rollout，重算 old_log_probs就行了，可选 ref/critic/reward，算 advantage，最后 update_actor。标准 PPO loss 是 ratio clipping:

$$
r_t = \exp(\log \pi_\theta(a_t) - \log \pi_{\text{old}}(a_t))
$$

$$
L = \max(-r_t A_t,\ -\operatorname{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)
$$

```python
# core_algos.py
pg_losses1 = -advantages * ratio
pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
```

- 直觉是：如果 $A_t>0$，这个 token 比较好，模型会想增大它的概率，但 $r_t$ 超过 $1+\epsilon$ 后就不再给额外收益；如果 $A_t<0$，这个 token 比较差，模型会想降低它的概率，但 $r_t$ 低于 $1-\epsilon$ 后也不再允许继续从 loss 里获利。max 选的是“更保守、更大的 loss”，防止单步更新把策略推太远。
- 区别在于优化信号
    - PPO: adv_estimator=gae 时需要 critic value，advantage 来自 reward + value bootstrap；
    - GRPO: 不需要 critic，通常每个 prompt 采多条 response，在同组内按 outcome reward 做均值/方差归一化，再广播到 token；
        - PPO surrogate + group-relative advantage
    - OPD: reward 不是外部标量为主，而是 teacher 给的 token 级分布差异。若 use_task_rewards=False，普通 PPO/GRPO 的 task reward policy loss 会被置零，只保留 distillation term；
        - PPO trainer 基础设施 + teacher KL/logprob 监督

- k1 + use_policy_gradient=True
    - distillation_loss_mode=k1
    - use_policy_gradient=True
    - use_task_rewards=False
- $\delta_t(\theta)=\log\pi_\theta(Y_t\mid S_t)-\log p_T(Y_t\mid S_t).$
    - $A_t^{\mathrm{distill}}=-\operatorname{sg}\!\left(\operatorname{clip}(\delta_t)\right)$

```python
advantages=-distillation_losses.detach()
```

----

- 纯 OPD 配方甚至不使用任务 advantage
    - `use_task_rewards=False`

```python
policy_loss, policy_metrics = ppo_loss(...)
if not use_task_rewards:
    policy_loss = 0.0

policy_loss += distill_loss
```