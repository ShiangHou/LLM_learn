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

