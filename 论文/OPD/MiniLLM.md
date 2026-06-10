# Mini LLM On-policy distillation for LLM

## 一、问题的提出

[MiniLLM: On-Policy Distillation of Large Language Models](https://arxiv.org/pdf/2306.08543)

像标题说的一样，这篇论文讲的就是 如何从一个LLM中通过“on policy distillation”的方法，得到一个迷你LLM

望文生义，所以肯定是之前有过大到小的这种蒸馏的方法，那么作者提出了一个新的方法（或者改进了一个新的方法）

过去的方法叫做KD(knowledge distillation)



作者说，之前的都是黑盒蒸馏，即直接用API来去蒸馏，关于如何去白盒蒸馏目前依旧是under-explored

为什么要这么做，作者指出了，LLM是一个生成式的任务，而不是分类任务，分类任务的概率空间很少，而生成式的任务概率空间很大，小模型很难去学到

作者指出，假设输入的prompt是x，输出是y，student模型是$q_{\theta}(y|x)$ ,教师是$p(y|x)$ ,那么我们在SFT的时候就是最小化$KL[p||q_{\theta}]$

这种KL散度也叫做**forward KL**，目标是让q能够尽可能的覆盖p，即q去拟合P



作者提出，在分类任务中，这种是非常好的，因为空间概率有限，但在文本生成任务中，教师模型的输出空间极其复杂，包含的模式远多于容量有限的学生模型所能表达的模式 。强制拟合会导致学生模型分配极高的概率给毫无意义的文本组合

这个时候可以用**reverse KL**，也就是$KL[q_{\theta}||p]$这样

 

![image-20260607160716029](https://cdn.jsdelivr.net/gh/ShiangHou/markdown_image/img/image-20260607160716029.png)

这个图里面可以看出，蓝色的是target，如果我们用forward KL的化，比如分类任务，那么他就会尝试拟合所有的，如果我们用reverse的话，他就会采样出最高概率的，即避免了一些无关紧要的分布在生成式任务中带来的影响



## 二、核心方法

这样的话最小化的目标就是


$$
\arg \min_{\theta} \mathcal{L}(\theta) = \arg \min_{\theta} KL[q_{\theta}||p] = \arg \min_{\theta} [-\mathbb{E}_{x\sim p_{x}, y\sim q_{\theta}} \log \frac{p(y|x)}{q_{\theta}(y|x)}]
$$
这里最关键的是采样分布变了：不是从 teacher 采样，也不是从真实数据采样，而是从 **student 自己当前的策略 $q_\theta$** 采样。这就是 on-policy。student 先自己生成回答，然后 teacher 对 student 的生成轨迹逐 token 计算概率反馈



这个和之前说的解决了两个问题， 一个是SFT的时候变成了on-policy的，另一个是RL时候的稀疏奖励，这里以teacher的logits的每一个token来作为奖励，避免了稀疏的奖励



下面是具体的公式细节

首先给定一个prompt输入的x，student会生成一序列的
$$
y = \{y_t\}_{t = 1}^{T}
$$


对于每一个token，会有一个奖励是
$$
r_t = \log \frac{p(y_t|y_{<t},x)}{q_{\theta}(y_t|y_{<t},x)}
$$
很显然，这是一个step level的奖励，对于每一步生成的都会有一个奖励

我们定义，从当前的位置t到结尾都有一个累加reward

从 $t$ 到 $T$ 的累加定义为累积奖励 $R_t$
$$
R_t = \sum_{t'=t}^T \log \frac{p(y_{t'}|y_{<t'}, x)}{q_{\theta}(y_{t'}|y_{<t'}, x)}
$$
回到我们的KL散度中，我们的目标函数是尽可能的让着reverse KL小，即
$$
\mathcal{L}(\theta) = KL[q_{\theta}||p] = -\mathbb{E}_{x\sim p_x, y\sim q_{\theta}(\cdot|x)} \log \frac{p(y|x)}{q_{\theta}(y|x)}
$$
我们对目标函数求导，可以得到
$$
\nabla \mathcal{L}(\theta) = -\int_{x,y} p_x(x) \nabla_{\theta} \left[ q_{\theta}(y|x) \log \frac{p(y|x)}{q_{\theta}(y|x)} \right] dx dy
$$
利用乘积法则 $\nabla(uv) = u\nabla v + v\nabla u$，我们将中括号内的项展开：
$$
= -\int_{x,y} p_x(x) \left[ q_{\theta}(y|x) \nabla_{\theta} \log \frac{p(y|x)}{q_{\theta}(y|x)} + \log \frac{p(y|x)}{q_{\theta}(y|x)} \nabla_{\theta} q_{\theta}(y|x) \right] dx dy
$$
因为 Teacher 模型的概率 $p(y|x)$ 是固定的常量，对 $\theta$ 求导为 0，所以：
$$
\nabla_{\theta} \log \frac{p(y|x)}{q_{\theta}(y|x)} = \nabla_{\theta} (\log p(y|x) - \log q_{\theta}(y|x)) = -\nabla_{\theta} \log q_{\theta}(y|x)
$$
同时，再结合对数导数技巧：$\nabla q_{\theta} = q_{\theta} \nabla \log q_{\theta}$，我们可以发现积分其实是对概率密度的梯度进行积分，结果为 0：
$$
\int q_{\theta}(y|x) (-\nabla_{\theta} \log q_{\theta}(y|x)) dy = -\int \nabla_{\theta} q_{\theta}(y|x) dy = -\nabla_{\theta} \int q_{\theta}(y|x) dy = -\nabla_{\theta}(1) = 0
$$
**对于展开后的第二项**： 同样应用对数导数技巧 $\nabla q_{\theta} = q_{\theta} \nabla \log q_{\theta}$，将其改写为期望的形式：
$$
-\int p_x(x) q_{\theta}(y|x) \log \frac{p(y|x)}{q_{\theta}(y|x)} \nabla_{\theta} \log q_{\theta}(y|x) dx dy
$$
将两者合并，我们可以得到一个统一的期望表达式：
$$
\nabla \mathcal{L}(\theta) = -\mathbb{E}_{x\sim p_x, y\sim q_{\theta}(\cdot|x)} \left[ \left(\log \frac{p(y|x)}{q_{\theta}(y|x)} - 1 \right) \nabla_{\theta} \log q_{\theta}(y|x) \right]
$$
回到我们的token序列中，对于自回归生成的文本，整句的概率是每个 step Token 概率的连乘,取对数后变成相加
$$
\log \frac{p(y|x)}{q_{\theta}(y|x)} = \sum_{t'=1}^T \log \frac{p(y_{t'}|y_{<t'}, x)}{q_{\theta}(y_{t'}|y_{<t'}, x)}
$$

$$
\nabla_{\theta} \log q_{\theta}(y|x) = \sum_{t=1}^T \nabla_{\theta} \log q_{\theta}(y_t|y_{<t}, x)
$$

$$
\nabla \mathcal{L}(\theta) = -\mathbb{E} \sum_{t=1}^T \left[ \left( \sum_{t'=1}^T \log \frac{p(y_{t'}|y_{<t'}, x)}{q_{\theta}(y_{t'}|y_{<t'}, x)} - 1 \right) \nabla_{\theta} \log q_{\theta}(y_t|y_{<t}, x) \right]
$$

在序列决策中，当前时刻 $t$ 的动作（生成 $y_t$）只会影响未来的 Reward，**不会影响过去的 Reward**。从数学角度来说，任何 $t' < t$ 的历史 Reward 项与当前步 $\nabla \log q_{\theta}(y_t)$ 的乘积，在当前策略分布下期望严格为 0 。  

因此，我们可以安全地将内部求和的起始点从 $1$ 截断到 $t$,这样就只剩下了
$$
\nabla \mathcal{L}(\theta) = -\mathbb{E} \sum_{t=1}^T \left[ \left( \sum_{t'=t}^T \log \frac{p(y_{t'}|y_{<t'}, x)}{q_{\theta}(y_{t'}|y_{<t'}, x)} - 1 \right) \nabla_{\theta} \log q_{\theta}(y_t|y_{<t}, x) \right]
$$
最后把刚才的R带回去，R就是$$R_t = \sum_{t'=t}^T \log \frac{p(y_{t'}|y_{<t'}, x)}{q_{\theta}(y_{t'}|y_{<t'}, x)}$$

最后就是
$$
\nabla \mathcal{L}(\theta) = -\mathbb{E}_{x\sim p_x, y\sim q_{\theta}(\cdot|x)} \sum_{t=1}^T (R_t - 1) \nabla \log q_{\theta}(y_t|y_{<t}, x)
$$
![image-20260607172016159](https://cdn.jsdelivr.net/gh/ShiangHou/markdown_image/img/image-20260607172016159.png)

不难发现， MiniLLM 实际上是一个 policy gradient 形式：student 自己 rollout，然后根据 teacher/student 的 log-prob ratio 得到 reward若某段生成在 teacher 下概率高、在 student 下还没有被充分学会，那么 $R_t$ 较高，训练会提升这些 token 的概率

但是作者指出，直接用会有问题，

1. policy gradient 方差高；
2. student 可能 reward hacking，生成重复、退化文本但获得高分；
3. 累计 reward 会偏向短输出，导致模型输出过短甚至空回答。

## 三、三个技巧

### 1：Single-Step Decomposition

这个解决的是第一个，即方差高，

分成两个部分，作者将单步生成质量 $r_t$ 从累积奖励 $R_t$ 中分解出来，直接计算单步生成质量的梯度期望 。由于单步质量可以遍历整个词表（Vocabulary）直接精确求和，而不是依赖蒙特卡洛采样，这大大降低了训练的方差并加速了收敛 。
$$
\nabla\mathcal{L}(\theta) = \mathbb{E}_{y\sim q_{\theta}(\cdot|x)}\left[-\sum_{t=1}^{T}\nabla_{y_{t}\sim q_{\theta}(t)}[r_{t}]\right] + \mathbb{E}_{y\sim q_{\theta}(\cdot|x)}\left[-\sum_{t=1}^{T}R_{t+1}\nabla \log q_{\theta}(y_{t}|y_{<t},x)\right]
$$
其中$r_t = \log \frac{p(y_t|y_{<t},x)}{q_{\theta}(y_t|y_{<t},x)}$表示了单步的质量

> 这个思想跟RL里面很想，就是当前的值可以直接获得，之前在做作业的时候MC确实不好估计，这里抠出来了第一个，按照广义的，也可也扣出来前t个，

### 2：Teacher-Mixed Sampling

如果完全从 student 采样，尤其是小 student 初期质量差，可能生成一些重复、无意义、退化文本，但这些文本在某些局部 token 上又能骗过 teacher，导致 reward hacking。

所以作者不是直接从 $q_\theta$ 采样，而是从 teacher 和 student 的混合分布采样：
$$
\tilde{p}(y_{t}|y_{<t},x) = \alpha \cdot p(y_{t}|y_{<t},x) + (1-\alpha) \cdot q_{\theta}(y_{t}|y_{<t},x)
$$
如果是混合采样，自然会想到用重要性采样来去修正一下
$$
w_{t} \approx \frac{q_{\theta}(y_{t}|y_{<t},x)}{\tilde{p}(y_{t}|y_{<t},x)}
$$
与此同时，单步的和长步的就是
$$
(\nabla\mathcal{L})_{Single} = -\mathbb{E}_{x\sim p_{x}, y\sim \tilde{p}(\cdot|x)}\left[\sum_{t=1}^{T}w_{t}\nabla_{y_{t}\sim q_{\theta}(t)}[r_{t}]\right]
$$

$$
(\nabla\mathcal{L})_{Long} = -\mathbb{E}_{x\sim p_{x}, y\sim \tilde{p}(\cdot|x)}\left[\sum_{t=1}^{T}w_{t}R_{t+1}\nabla \log q_{\theta}(y_{t}|y_{<t},x)\right]
$$

> 这有点像 RL 里面为了稳定训练做 off-policy / importance weight 修正

### 3:Length Normalization

作者观察到，原始累计 reward $R_{t+1}$ 会偏向短句子，导致模型倾向于输出短回答。因此他们对未来 reward 做长度归一化, 对累加分数 $R_{t+1}$ 进行长度归一化（除以句子剩余的步数），从而消除长度带来的偏差 

归一化后的reward是
$$
R_{t+1}^{Norm} = \frac{1}{T-t-1}\sum_{t'=t+1}^{T} \log \frac{p(y_{t'}|y_{<t'},x)}{q_{\theta}(y_{t'}|y_{<t'},x)}
$$
它的作用是把“未来累计 reward”变成“未来平均 reward”，避免长回答因为 token 多而被惩罚

最后就是
$$
\nabla\mathcal{L}(\theta) = -\mathbb{E}_{x\sim p_{x}, y\sim \tilde{p}} \sum_{t=1}^{T} \frac{q_{\theta}(y_{t}|y_{<t},x)}{\tilde{p}(y_{t}|y_{<t},x)} \left[ R_{t+1}^{Norm} \nabla \log q_{\theta}(y_{t}|y_{<t},x) + \nabla \sum_{y'_{t} \in V} q_{\theta}(y'_{t}|y_{<t},x) \log \frac{p(y'_{t}|y_{<t},x)}{q_{\theta}(y'_{t}|y_{<t},x)} \right]
$$
论文里还加入了 PPO 类似的 clipping 策略，并加入一个普通语言模型损失：
$$
\mathcal{L}_{PT}
=
-
\mathbb{E}_{d \sim D_{PT}}
\log q_\theta(d)
$$
这个 $\mathcal{L}_{PT}$ 用来保持模型的基础语言能力，避免模型只在 instruction 数据上过拟合。

## 四、训练流程

MiniLLM 的训练流程分两阶段。

第一阶段，先对 student 做普通 SFT。用 instruction-response 数据 $D$ 训练 student，并选择 validation loss 最低的 checkpoint 作为后续 on-policy 蒸馏的初始化点。

第二阶段，进行 MiniLLM 训练：

1. 从训练集 $D$ 采样 prompt；
2. 用 teacher-student 混合分布 $p_e$ 生成回答；
3. teacher 和 student 都对这条生成轨迹计算 token-level probability；
4. 计算 single-step gradient；
5. 计算 length-normalized long-term gradient；
6. 从预训练语料 $D_{PT}$ 采样文本，计算语言模型损失；
7. 用三部分梯度更新 student：

$$
(\nabla \mathcal{L})_{\text{Single}}
+
(\nabla \mathcal{L})_{\text{Long}}^{\text{Norm}}
+
\nabla \mathcal{L}_{PT}
$$

论文自己也说，这个训练管线和 RLHF 很像：都是先有一个 SFT 初始模型，然后用 policy optimization 继续训练。区别是 MiniLLM 的“奖励”不是人类反馈模型，而是 teacher LLM 的 token-level 分布。





## 实验

作者把任务设为 instruction-following。训练数据来自 **Databricks Dolly 15K**，过滤超长样本后，划分出 1K validation、0.5K test，剩下约 12.5K 训练样本。预训练语料 $D_{PT}$ 方面，GPT-2 系列用 OpenWebText，其他模型用 RoBERTa training corpus。teacher-mix-in 强度 $\alpha=0.2$。

模型规模覆盖比较广：

| Teacher    | Student                  |
| ---------- | ------------------------ |
| GPT-2 1.5B | GPT-2 120M / 340M / 760M |
| OPT 13B    | OPT 1.3B / 2.7B / 6.7B   |
| LLaMA 13B  | LLaMA 7B                 |
| GPT-J 6B   | 附录中补充实验           |

可以看到说，teacher模型也都是十几B的小模型。在qwen的技术报告里面，qwen系类的模型都是最大参数模型OPD蒸馏来的，这样的

## 总结

创新点主要有四个。

第一，**把 LLM 蒸馏问题重新表述为 reverse KL 优化问题**。过去 KD 默认用 forward KL，作者指出这对开放式生成不合适，因为小模型无法覆盖 teacher 的复杂多模态分布。

第二，**提出 on-policy distillation**。SeqKD 是 teacher 生成，student 模仿；MiniLLM 是 student 自己生成，teacher 反馈。这个转变很关键，因为它缓解了训练-推理不一致，也就是 exposure bias。

第三，**把 teacher 的 token-level probability 当成隐式奖励**。不需要人工 reward model，也不需要偏好数据。只要 teacher 白盒可访问，就可以通过：
$$
\log \frac{p(y_t|y_{<t},x)}{q_\theta(y_t|y_{<t},x)}
$$
构造训练信号。

第四，**提出了一套让 reverse KL policy optimization 可训练的工程技巧**：single-step decomposition、teacher-mixed sampling、length normalization、PPO-style clipping、额外 LM loss。没有这些技巧，直接优化 reverse KL 很容易高方差、reward hacking 或输出过短。