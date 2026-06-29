### GKD

## 一、问题的提出 

[[2306.13649\] On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://arxiv.org/abs/2306.13649)

依旧继续望文生义，标题翻译一下就是，从模型的self-Generated Mistakes中，设计的一种on-policy的知识蒸馏，

作者提出了一种General Knowledge Distillation的方法

注意一下，有一个骚猪论文的名字里面就叫GKD，但不是这个方法。这里的这个方法才是正宗的

论文里面依旧是提到了两个问题，即传统Knowledge Distillation的问题

**一、训练和推理之间存在分布不匹配**

自回归语言模型的生成过程为
$$
p_\theta(y\mid x)
=
\prod_{t=1}^{L_y}
p_\theta(y_t\mid x,y_{<t})
$$
这个x就是prompt，即输入，y就是输出，然后后面的t就是一大长串的token，连乘就是表示这个极大似然估计嘛，所以这是一个

即：

- $x$ 是输入；
- $y_t$ 是第 $t$ 个输出 token；
- $y_{<t}$ 是此前已经生成的前缀。

传统 SFT 或知识蒸馏训练时，模型通常看到的是data的数据，这些数据不是自己产生的，而是来自教师或者训练数据

但是再自己推理的时候，学生模型看到的是自己生成的前缀：
$$
y_{<t}^{S}\sim p_S
$$
因此训练和推理所处的状态分布不同：
$$
d_{\text{train}}(x,y_{<t})
\neq
d_{p_S}(x,y_{<t})
$$


这种问题通常称为：

- exposure bias，暴露偏差；
- train-inference distribution mismatch，训练—推理分布不匹配；
- imitation learning 中的 covariate shift，状态分布偏移。

自回归生成中，早期一个 token 出错，会改变后面所有 token 的条件分布，所以误差可能级联放大。

我们在MiniLLM里面也说了这一点，就是所谓的exposure bias

**二、学生不一定有能力完整拟合教师分布**

传统知识蒸馏主要使用前向 KL：
$$
D_{\mathrm{KL}}(p_T\Vert p_S)
$$
它要求学生尽量覆盖教师的所有概率模式。

但是教师模型可能有 30 亿参数，而学生只有 7700 万参数。学生表达能力不足时，强迫它覆盖教师的全部输出分布，可能导致它：

- 在多个模式之间平均；
- 给教师低概率 token 分配不必要的概率；
- 产生看似多样、实际质量不高的输出；
- 在高温采样时更容易输出低质量内容。

因此论文的第二个问题是：

> 当教师和学生容量差距较大时，是否应该始终使用前向 KL？

作者的答案是否定的。不同任务可能适合前向 KL、反向 KL 或不同参数的 JSD。

> 补充一下，这里和MiniLLM的思想是一样的，即对于分类任务（比如意图识别等，还是要用forward KL，但是对于生成任务，还是需要反向reward

## 二、之前的方法

这里介绍了几个之前的方法，主要是几种，分别是SFT、Supervised KD、SeqKD

### SFT

也就是Supervised Fine-Tuning，监督微调，这里就是给定一个数据

给定人工数据集 $(X,Y)$，SFT 最小化真实答案的负对数似然：
$$
\mathcal L_{\mathrm{SFT}}(\theta)
=
\mathbb E_{(x,y)\sim(X,Y)}
\left[
-\log p_S^\theta(y\mid x)
\right]
$$
展开到 token 级别：
$$
\mathcal L_{\mathrm{SFT}}
=
-\mathbb E_{(x,y)}
\left[
\sum_{t=1}^{L_y}
\log p_S^\theta(y_t\mid x,y_{<t})
\right]
$$
它相当于使用 one-hot 标签：
$$
q(v)=
\begin{cases}
1,&v=y_t\\
0,&v\neq y_t
\end{cases}
$$
SFT是老生常谈的方法，SFT本质是KL散度的one-hot形式，相当于是要完全拟合我们的数据集



### Sequence-level KD



这个是序列级别的knowledge distillation

SeqKD 先让教师生成一条答案：
$$
\hat y_T\sim p_T(\cdot\mid x)
$$
然后将教师答案当作普通 SFT 数据：
$$
\mathcal L_{\mathrm{SeqKD}}
=
-\mathbb E_x
\left[
\log p_S^\theta(\hat y_T\mid x)
\right]
$$
和SFT的loss一摸一样，唯一的区别在于是这个序列是另一个模型生成的，仅此而已，可能是比人工的更科学

###  Supervised KD

如果我们拿到了teacher在每一个状态下的完整的token概率分布，那么就不用one-hot了



监督蒸馏不仅使用教师生成的 token，还使用教师的完整 token 概率分布。

在状态
$$
s_t=(x,y_{<t})
$$
下，记：
$$
P_t(v)=p_T(v\mid s_t),\qquad
Q_t^\theta(v)=p_S^\theta(v\mid s_t)
$$
则前向 KL 为
$$
D_{\mathrm{KL}}(P_t\Vert Q_t^\theta)
=
\sum_{v\in V}
P_t(v)
\log
\frac{P_t(v)}{Q_t^\theta(v)}
$$
整条序列的 token 平均蒸馏损失为：
$$
D(P_T\Vert P_S^\theta)(y\mid x)
=
\frac{1}{L_y}
\sum_{t=1}^{L_y}
D\left(
p_T(\cdot\mid x,y_{<t})
\Vert
p_S^\theta(\cdot\mid x,y_{<t})
\right)
$$
监督 KD 的目标是：
$$
\mathcal L_{\mathrm{SD}}(\theta)
=
\mathbb E_{(x,y)\sim(X,Y)}
\left[
D_{\mathrm{KL}}(P_T\Vert P_S^\theta)(y\mid x)
\right]
$$
它比 SFT 的信号更丰富，因为每个位置都有完整的教师概率分布。

但是这三种其实都是off-policy的，即数据都是别人给的，不是student自己生成的，所以会有刚才说的一些问题



## 三、GKD思想

用所谓的对比学习，即

将：

- 学生模型视为 policy；
- 教师模型视为 interactive expert；
- 前缀 $y_{ 视为 state；
- 下一个 token 视为 action。

传统蒸馏是在专家或固定数据访问的状态上训练学生。

GKD 则采用类似 DAgger 的思路：

1. 学生访问自己真实会访问的状态；
2. 教师在这些状态上给出专家分布；
3. 学生学习教师在这些状态下的行为。

对输入 $x$，首先让学生生成：
$$
y\sim p_S^\theta(\cdot\mid x)
$$
然后在学生生成的每一个前缀
$$
(x,y_{<t})
$$
上，同时运行教师和学生，得到：
$$
p_T(\cdot\mid x,y_{<t})
$$
和
$$
p_S^\theta(\cdot\mid x,y_{<t})
$$
用两者的 token 分布差异训练学生。

On-policy KD 的目标为：
$$
\mathcal L_{\mathrm{OD}}(\theta)
=
\mathbb E_{x\sim X}
\left[
\mathbb E_{y\sim p_S(\cdot\mid x)}
\left[
D_{\mathrm{KL}}
\left(
p_T\Vert p_S^\theta
\right)(y\mid x)
\right]
\right]
$$
与Supervised KD 的关键区别仅在于 $y$ 的来源：
$$
\begin{aligned}
\text{Supervised KD:}\quad
&y\sim p_{\mathrm{data}}
\\
\text{On-policy KD:}\quad
&y\sim p_S
\end{aligned}
$$
这里有人可能就要问了，我的模型在训练的时候，比如每一次更新了参数，那么下一次的训练需不需要用这个新模型，即

> 学生生成了哪些 token，本身取决于学生参数 $\theta$。在计算梯度时，要不要考虑“参数改变后，学生采样到这些 token 的概率也会改变”。

GKD 的做法是：**不考虑这一层依赖关系**。它先让学生生成一条序列，然后把这条序列当作固定数据，只在这条序列对应的状态上计算教师—学生分布差异。