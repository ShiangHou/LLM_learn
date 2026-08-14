# 一、项目整体

### Q1：先介绍一下你的 LLM Rerank 项目。

**答：**

这个项目是在 AI 搜索多工具并行召回之后做统一结果融合。上游不同工具，比如产品搜索、RAG 等，每一路都会返回几十个候选产品及相关 chunk，合并去重以后一个 Query 大概有上百个候选。

最开始我们使用 RRF 做多路召回融合，但是 RRF 主要依赖各召回链路中的排名，无法真正理解 Query 和产品、chunk evidence 之间的语义相关性，而且最终业务不仅需要 Top20，还要求每个产品给出推荐理由。

所以我基于 Qwen3-30B-A3B 做了一个生成式 Listwise Reranker，输入 Query、候选产品及对应 evidence，直接输出 Top20 和每个产品的理由。

训练上采用两阶段 SFT：第一阶段只学习候选筛选和排序，第二阶段在已有排序能力基础上联合训练排序和理由生成。之后针对 NDCG 较低、Top20 多次采样不稳定、RRF 与 Gold 分歧明显以及 Top20 边界 Hard Negative 的 Query 做 GSPO，reward 主要由 NDCG@20、Recall@20、理由质量和格式约束组成。

------

### Q2：为什么需要 LLM Rerank？RRF 不够吗？

**答：**

RRF 的优势是不同召回系统的 score 不需要做 calibration，它直接利用 rank 做融合：

Score(d)=m∑k+rankm(d)1

但它的问题是没有真正建模：

Query↔Candidate↔Evidence

之间的语义关系。

比如一个产品因为三个召回器都排得比较靠前，所以 RRF 分数很高，但它可能并不满足 Query 中的核心约束。

另外我们的业务还要求生成理由，因此使用 LLM 可以把：

Ranking+Explanation

统一在一个模型里。

------

### Q3：那为什么不用 Cross Encoder + Generator？

**答：**

这是一个合理的方案，而且如果线上 latency 要求特别高，我也会考虑 cascade：

Retriever→CrossEncoder→LLM

我们这里选择生成式 LLM Reranker，主要因为候选包含比较丰富的自然语言 chunk，并且最终一定要生成理由。Listwise LLM 可以同时看到多个候选，利用候选之间的相对关系进行排序，同时完成 explanation。

它的代价是推理成本更高、长候选情况下稳定性下降，所以我们不会让 LLM 面对全库，而是在召回之后的有限候选池上做 rerank。

------

# 二、RRF

### Q4：RRF 具体怎么算？

假设候选 d 在第 m 路召回中的排名是：

rankm(d)

则：

RRF(d)=m=1∑Mk+rankm(d)1

例如两路结果：

```
Tool1:
A rank1
B rank5

Tool2:
A rank10
B rank2
```

那么：

RRF(A)=k+11+k+101RRF(B)=k+51+k+21

然后按照 RRF score 排序。

------

### Q5：为什么 RRF 不直接使用各召回模型的分数？

因为不同召回模型的 score 没有统一尺度。

比如：

BM25=15.2EmbeddingSimilarity=0.82RuleScore=100

直接相加没有意义。

RRF 把它们统一转换成：

> 第几名

从而避免 score calibration。

------

### Q6：RRF 最大缺陷是什么？

**答三点：**

1. 不理解 Query 和 Candidate 的真实语义。
2. 基本不利用 chunk 中的复杂 evidence。
3. 只能融合排序，不能完成 explanation。

------

# 三、为什么是 Listwise

### Q7：Pointwise、Pairwise、Listwise 有什么区别？

**Pointwise：**

单独预测：

si=f(q,di)

然后对所有 si 排序。

------

**Pairwise：**

学习：

di>dj

比如 RankNet。

------

**Listwise：**

直接输入：

q,[d1,⋯,dN]

输出完整排序或者 TopK：

[dπ1,...,dπK]

我们的业务本身就是：

> 从一批候选里面选择 Top20。

因此 Listwise 和最终业务目标更一致。

------

### Q8：Listwise 就一定比 Pointwise 强吗？

不一定。

Listwise 的优势是可以利用候选之间的相对信息，但缺点是：

- 输入很长；
- 推理贵；
- 容易受到 candidate position bias；
- candidate 数量增加后质量容易下降。

所以最终还是需要实验验证。

------

# 四、数据和 Gold

### Q9：训练数据是怎么构造的？

**答：**

整体流程是：

Query→多工具实际召回→候选合并去重→Query−Candidate相关性标注→GoldTop20→Reason标注

其中重点会保留：

- 普通正样本；
- Easy Negative；
- Hard Negative；
- Top20 cutoff 附近的 Boundary Samples。

尤其 RL 阶段重点使用 Hard Query 和 Boundary Query。

------

### Q10：你说的相关性标签怎么来的？

这是人工/业务标注体系得到的 **graded relevance**。

例如：

rel∈{0,1,2,3}

可以定义成：

- 3：强相关，直接满足 Query 核心需求；
- 2：相关，基本满足；
- 1：弱相关，只部分满足；
- 0：不相关或违反核心约束。

例如 Query：

> 低风险、短期闲钱、流动性好

可能：

```
货币基金       rel=3
短债基金       rel=3
中长期债基     rel=1/2
股票基金       rel=0
```

这个 relevance 后续用于计算 NDCG。

------

### Q11：所以 Gold 是一个完整的唯一排序吗？

**不是。**

这个问题很关键。

我们真正标注的是：

rel(q,d)

而不是要求人工严格规定：

A>B>C>D

例如：

```
A rel=3
B rel=3
C rel=2
```

A 和 B 谁在前面，业务意义上可能都正确。

因此 NDCG 比 exact permutation match 更合理。

------

### Q12：为什么不用 LLM 自动标所有 relevance？

可以用强模型辅助生产 pseudo label，但是最终高质量 validation/test set 最好人工确认。

否则可能变成：

> 用 Judge 的偏好训练模型，再用同一个 Judge 的偏好评估模型。

这样测到的更多是模型和 Judge 的一致程度，而不是真实业务 relevance。

------

### Q13：Hard Negative 是什么？

Hard Negative 是：

> 表面上和 Query 很像，但是核心意图不匹配的候选。

例如 Query：

> 沪深300ETF

候选：

```
沪深300ETF            positive
沪深300ETF联接        related
沪深300增强基金       hard case
中证500ETF            hard negative
股票主动基金           easy negative
```

真正提升 rerank 决策边界的，往往是 Hard Negative，而不是 Easy Negative。

------

### Q14：为什么特别关注 Top20 boundary？

因为真正影响：

Recall@20

和：

NDCG@20

的不是 rank1 和 rank150 的区别。

而是：

rank18,19,20,21,22

之间谁进去、谁被挤出去。

所以 TopK ranking 本质上特别依赖 cutoff 附近的 decision boundary。

------

# 五、NDCG

### Q15：NDCG@20 怎么算？

首先根据模型输出的 Top20 candidate ID 查 Gold relevance：

rel1,rel2,...,rel20

计算：

DCG@20=i=1∑20log2(i+1)2reli−1

然后把候选按照 Gold relevance 从高到低排序，得到理想排序：

IDCG@20

最终：

NDCG@20=IDCG@20DCG@20

------

### Q16：为什么是 2rel−1？

为了让不同 relevance level 的差距具有非线性。

例如：

rel=3⇒gain=7rel=2⇒gain=3rel=1⇒gain=1

因此把强相关产品排在前面的收益会明显更高。

------

### Q17：为什么要除以 log2(i+1)？

因为用户对于高排名结果更敏感。

rank1 和 rank20 即使相关性一样，对用户体验的贡献也不同。

所以：

log2(i+1)1

是 position discount。

------

### Q18：为什么还需要 IDCG？

因为不同 Query 的候选池难度不一样。

有的 Query 有 20 个强相关产品，有的可能只有 3 个。

单纯 DCG 不可直接跨 Query 比较。

所以：

NDCG=IDCGDCG

归一化到：

[0,1]

方便跨 Query 聚合。

------

### Q19：NDCG 和 Recall@20 有什么区别？

Recall@20：

Recall@20=候选池中的Relevant总数量Top20中的Relevant数量

主要看：

> 好产品有没有被召回来。

NDCG 还考虑：

> 好产品在 Top20 里面排在哪里。

例如两个模型 Recall 都是 1，但一个把强相关产品放 rank1，一个放 rank20，NDCG 会明显不同。

------

### Q20：为什么不用 MRR？

MRR：

MRR=∣Q∣1q∑rankq1

主要关心：

> 第一个 relevant item 出现在第几位。

而我们需要整个 Top20 高质量，因此 NDCG@20 更符合业务目标。

------

# 六、两阶段 SFT

### Q21：为什么要做两阶段 SFT？

这是核心题。

最终任务其实是两个任务：

Ranking

和：

ReasonGeneration

如果直接联合训练，理由部分通常有几百甚至上千 token，而 ranking ID 可能只有几十个 token。

普通 CE：

L=−t∑logp(yt)

每个 token 都产生梯度。

因此理由生成很容易在 token 数量上压过排序信号。

所以我们使用 curriculum learning：

第一阶段：

> 只学选谁和怎么排。

第二阶段：

> 在已经有 ranking initialization 的情况下，再学 reason。

------

### Q22：Stage1 输入输出是什么？

输入：

```
Query
Candidate1 + evidence
Candidate2 + evidence
...
CandidateN + evidence
```

输出只包含：

```
C017
C082
C006
...
```

即 Top20 candidate list。

不生成 reason。

------

### Q23：Stage1 用什么 Loss？

没有额外 ranking head。

还是标准 causal LM Cross Entropy：

L=−T1t∑logπθ(yt∣x,y<t)

本质是：

> 把 Listwise ranking 序列化成语言模型 response。

------

### Q24：普通 next-token prediction 怎么学排序？

因为完整排序：

[C17,C82,C6,...]

本身被序列化成一个 target sequence。

模型学习：

P(C17,C82,C6,...∣Query,Candidates)

利用概率链式法则：

P(y∣x)=t∏P(yt∣x,y<t)

所以 ranking 被转换成 conditional generation。

------

### Q25：Stage2 怎么训练？

使用 Stage1 checkpoint 初始化。

输入基本不变，但是输出：

```
{
  "results": [
    {
      "candidate_id": "C017",
      "reason": "..."
    }
  ]
}
```

同时学习：

> ranking + reason generation。

------

### Q26：两阶段训练是不是线上要跑两次模型？

不是。

这是：

> 两阶段训练。

最终只有 Stage2 checkpoint。

线上：

Query+Candidates→FinalModel→Top20+Reasons

只 forward 一次。

------

# 七、SFT 的问题与 RL 动机

### Q27：SFT 已经有 Gold 了，为什么还需要 RL？

SFT 优化：

maxlogP(ygold∣x)

但是实际业务指标：

NDCG@20

两者存在 objective mismatch。

例如：

```
A rel=3
B rel=3
C rel=2
```

Gold sequence：

```
A,B,C
```

模型输出：

```
B,A,C
```

从 SFT CE 来看：

> 和 teacher sequence 不一样，要惩罚。

但：

NDCG=1

因为 A 和 B relevance 一样。

所以 RL 可以直接优化：

E[R]

而不是要求复刻唯一序列。

------

### Q28：为什么不是所有数据都拿来 RL？

大量简单 Query：

- SFT 已经做得很好；
- 多次 rollout reward 都一样；
- advantage 接近 0。

继续 RL 学习价值很低。

所以重点挖：

1. NDCG 低；
2. 多次 Top20 不稳定；
3. RRF 与 Gold 分歧；
4. Boundary Query；
5. Hard Negative 密集的 Query。

------

# 八、Stability

### Q29：Top20 Stability 怎么算？

同一个 Query rollout G 次，得到：

T1,T2,...,TG

两两计算 overlap：

Overlap(Ti,Tj)=20∣Ti∩Tj∣

平均：

Stability@20=G(G−1)2i<j∑Overlap(Ti,Tj)

越低说明模型对于 Top20 cutoff 的决策越不稳定。

------

### Q30：线上 temperature=0，为什么还看 sampling stability？

因为这里 sampling 不是为了模拟线上随机生成，而是：

> 用来估计模型 uncertainty。

如果 rank20、21、22 的概率很接近，小扰动就会导致结果变化。

因此低 Stability 往往意味着：

> decision margin 比较小，是高价值 RL Query。

------

# 九、GSPO

### Q31：为什么使用 GSPO？

三个理由：

第一，我们最终 reward 是完整 Top20 的 NDCG，本身就是 sequence-level reward。

第二，GSPO 使用 sequence-level importance ratio，优化粒度和完整 ranking response 更匹配。

第三，backbone 是 MoE，因此在工程上也希望使用 sequence-level policy optimization 来提高训练稳定性。

但最终是否优于 GRPO，还是需要通过 ablation 验证。

------

### Q32：GSPO 和 GRPO 最大区别是什么？

GRPO 主要使用 token-level importance ratio：

ri,t=πold(yi,t∣x,yi,<t)πθ(yi,t∣x,yi,<t)

GSPO 将整条 sequence 聚合：

si=exp[T1t∑logπold(yi,t)πθ(yi,t)]

一句话：

> **GRPO 更偏 token-level ratio，GSPO 使用 sequence-level ratio。**

------

### Q33：为什么 GSPO 要除以 sequence length？

因为：

P(y)=t∏P(yt)

如果直接计算整条 sequence probability ratio：

t∏rt

序列越长数值方差越大。

于是取：

(t∏rt)1/T

也就是 geometric mean。

在 log space：

T1t∑logrt

这样不同 response length 更可比较，也更稳定。

------

### Q34：GSPO Advantage 怎么计算？

同一个 Query sample G 个 response：

y1,...,yG

得到：

R1,...,RG

计算 group mean：

μ=G1i∑Ri

std：

σ=G1i∑(Ri−μ)2

然后：

Ai=σ+ϵRi−μ

高于组内平均：

Ai>0

低于组内平均：

Ai<0

------

### Q35：为什么不需要 Critic？

PPO 中 Critic 学习：

V(s)

作为 baseline。

GSPO 利用：

> 同一个 Query 的其他 rollout 的 reward

作为相对 baseline。

因此可以直接计算 group-relative advantage，不需要单独训练 value model。

------

### Q36：GSPO Loss 是什么？

核心形式：

J=G1i∑min[siAi,clip(si,1−ϵ,1+ϵ)Ai]

实际训练最小化：

L=−J

本质和 PPO clipping 一样：

> 好样本增加概率，坏样本降低概率，但每次 policy 更新不能过大。

------

### Q37：为什么需要 Importance Sampling Ratio？

因为 rollout 数据是旧 policy：

πold

采样出来的。

而真正更新参数的是当前：

πθ

所以需要：

πoldπθ

来校正两者的 policy distribution difference。

------

### Q38：为什么要 Clip？

如果某条偶然拿高 reward 的 sequence：

ratio≫1

一次 update 可能让它概率暴涨，造成 policy collapse。

所以：

clip(r,1−ϵ,1+ϵ)

限制单次更新。

本质上是：

> approximate trust region。

------

### Q39：如果一个 group 的 reward 全一样怎么办？

如果：

R1=R2=⋯=RG

那么：

Ai≈0

基本没有 policy gradient。

这也是为什么我们重点选择 hard queries，并且 rollout 时保留一定 sampling diversity。

简单 Query 没有 reward variance，本身也不适合拿来做大量 RL。

------

# 十、Reward

### Q40：Reward 怎么设计？

可以写：

R=αRrank+βRreason+γRformat−Pinvalid

其中最重要的是 ranking reward。

比如：

Rrank=0.8NDCG@20+0.2Recall@20

理由 reward 可以考虑：

- Groundedness；
- Query relevance；
- Conciseness。

格式部分可以检查：

- JSON 是否合法；
- 是否正好 20 个；
- ID 是否存在；
- 是否重复；
- rank 是否完整。

**实际权重一定以你最终实验为准，不要把示例权重说成真实线上配置。**

------

### Q41：为什么 NDCG 是主要 Reward？

因为我们的目标不仅是：

> 有没有选到 relevant 产品。

还要考虑：

> 强相关产品是不是排在更前面。

NDCG 同时包含 graded relevance 和 position discount，因此和 Top20 rerank 目标比较一致。

------

### Q42：Reason Reward 怎么算？

假设第 j 条理由：

rj

评分：

GroundednessjQueryRelevancejQualityj

可以定义：

Rreason=201j∑(w1Gj+w2Qj+w3Cj)

其中 deterministic 的部分尽量 rule-based。

比如：

> ID 是否存在、长度、格式、重复

不用 LLM Judge。

只有 groundedness、语义相关性这种难规则化的内容，再使用 Judge。

------

### Q43：为什么 Stability 不直接放大权重当 Reward？

因为可能 Reward Hacking。

模型如果每次都输出：

> 同一套错误 Top20

那么：

Stability=1

但 ranking 是错的。

所以 Stability 更适合作为：

> hard-query mining 和 evaluation metric。

而不是主要优化目标。

------

### Q44：有哪些 Reward Hacking？

至少说四个：

1. 只输出几个最确定的产品，提高表面 precision；
2. 重复同一个高相关产品；
3. reason 大量复制 evidence，提高 groundedness；
4. 使用万能模板理由，提高 Judge 分数。

所以必须加入：

- exact Top20；
- dedup；
- candidate existence；
- length；
- specificity；
- groundedness；

等规则约束。

------

# 十一、为什么不用 DPO / PPO

### Q45：为什么不用 DPO？

DPO 更适合已经有：

(yw,yl)

这种明确 preference pair 的场景。

而我们的 ranking task 可以直接得到：

NDCGRecall

等 scalar reward。

所以在线 rollout：

Policy→Response→Reward

更加自然，而且能持续挖掘当前 policy 的边界错误。

------

### Q46：为什么不用 PPO？

PPO 通常需要：

- Actor；
- Critic；
- Reward；
- Reference。

Critic 负责学习：

V(s)

而我们的 Query 可以直接采样多个 Top20 并互相比较，因此 group-relative baseline 已经能够提供 advantage。

不需要额外训练 critic，训练链路更简单。

------

# 十二、Qwen3-30B-A3B / MoE

### Q47：为什么选 Qwen3-30B-A3B？

主要考虑：

1. 输入包含大量 candidate + chunk，需要较强长上下文能力；
2. 最终既做 ranking 又生成 reason，需要比较好的生成能力；
3. MoE 模型总容量较大，但每个 token 只激活部分 experts，计算开销相比同规模 Dense 更可控。

------

### Q48：30B-A3B 是什么意思？

不是说：

> 模型只有 3B。

而是：

> 总参数量大约 30B，但是对于单个 token，只激活其中约几个 B 规模的专家参数。

因此：

TotalParams=ActiveParamsPerToken

------

### Q49：MoE 怎么工作的？

每个 token hidden state：

h

先经过 router：

p=softmax(Wrh)

然后选择 TopK experts：

Ei1,...,Eik

最终：

y=j∈TopK∑pjEj(h)

所以不同 token 可以被路由到不同专家。

------

### Q50：MoE 为什么还是很吃显存？

虽然每个 token 只激活部分专家：

> 计算量是稀疏的。

但是推理时通常仍然要把绝大多数甚至全部 expert 参数放进显存/设备中。

所以：

ComputeSparse

不等于：

MemorySparse

------

### Q51：MoE 有什么训练问题？

常见：

- expert load imbalance；
- expert collapse；
- router instability；
- all-to-all communication；
- 不同专家更新不均匀。

因此通常需要：

> load balancing 等机制。

------

# 十三、veRL 实现

### Q52：你在 veRL 里面一次 GSPO 训练大概怎么跑？

可以按照这个顺序回答：

```
1. DataLoader取一批Hard Query
2. Actor/Rollout policy用vLLM生成G条response
3. 保存old log probs
4. Reward function解析Top20
5. 算NDCG、Recall、Reason、Format reward
6. 同Query组内计算advantage
7. 当前Actor重新forward得到current log probs
8. 聚合成sequence-level ratio
9. 算GSPO clipped loss
10. backward + optimizer.step()
```

这就是一次 RL iteration 的主要流程。

------

### Q53：old policy 是什么？

每次 rollout 时使用的 policy：

πold

生成完这一批数据以后，它对应的 log probability 会固定下来。

训练 actor 的过程中：

πθ

不断变化。

于是：

πoldπθ

也不断变化。

等下一次重新 rollout，再重新确定新的：

πold

------

### Q54：为什么 rollout 和 training 通常要分开？

Rollout 更适合：

> 高吞吐 autoregressive inference，例如 vLLM。

Training 更适合：

> 保存 activation、计算梯度、backward，例如 FSDP/Megatron。

所以 RL Framework 会把：

Generation

和：

GradientUpdate

作为两个不同阶段优化。

------

# 十四、工程问题

### Q55：怎么防止输出不存在的 Candidate？

推荐四层：

1. 每个 Candidate 分配 query-local ID，例如 C001；
2. 输出只允许引用 candidate ID；
3. parser 检查 ID 是否存在；
4. RL 中加入 invalid-candidate penalty。

这是比让模型重新生成长基金名称更稳定的工程方案。

------

### Q56：怎么处理输入顺序偏置？

Listwise LLM 容易认为：

> 输入靠前的 candidate 更重要。

所以可以在数据构造时：

Shuffle(Candidates)

同一个 Query 多次构造不同输入 permutation，但 Gold 不变。

评测时也可以做：

> Candidate Shuffle Robustness Test。

------

### Q57：为什么候选变多以后模型会掉点？不是 Context Window 还没满吗？

这是很好的问题。

**能放进 context ≠ 能高质量排序。**

候选数增加以后：

- 语义竞争项增加；
- attention 更分散；
- 相似 candidate 更多；
- evidence 干扰更强；
- Top20 decision boundary 更复杂。

所以：

ContextCapacity=EffectiveRankingCapacity

即使 token 没超过窗口，也可能发生 ranking degradation。

------

# 十五、结果与实验

### Q58：你们怎么证明两阶段 SFT 有效？

一定准备 ablation：

```
Zero-shot
One-stage Joint SFT
Stage1 Ranking-only
Two-stage SFT
Two-stage SFT + GSPO
```

然后至少比较：

NDCG@20Recall@20ReasonPassRate

如果没有 One-stage baseline，面试时不要说：

> “实验证明两阶段一定更好。”

而应该说：

> 这是我们的设计动机，已有 Stage1→Stage2 的效果，但严格验证 curriculum 本身的贡献还需要 one-stage joint SFT ablation。

这种回答反而更可信。

------

### Q59：GSPO 提升到底来自哪里？

最好分 bucket 看：

```
Easy Query
Hard Query
Boundary Query
RRF-Gold disagreement
```

如果整体 NDCG 提升不大，但是：

HardQuery NDCG

明显提升，反而非常符合你的训练设计。

因为 GSPO 本来就不是重新学习整个任务：

> 是针对 SFT 决策边界做优化。

------

### Q60：你这个项目最大的不足是什么？

这是非常值得准备的收尾答案：

> 第一，Listwise LLM 的成本相对传统 reranker 更高，候选数过多以后 ranking quality 会下降，因此后续可以考虑 cascade 或 hierarchical reranking。
>
> 第二，离线 relevance 和 reason reward 的质量高度依赖标注体系，尤其 LLM Judge 需要人工 calibration。
>
> 第三，SFT 阶段依然使用 single-sequence CE，存在 gold permutation 和真实 ranking metric 不完全一致的问题，这也是我们引入 RL 的主要原因之一。
>
> 第四，GSPO 是 sequence-level credit assignment，对于具体哪个 candidate 导致 reward 下降无法精确定位，这是 sequence-level RL 本身的局限。

------

# 最后，你现在至少要做到这 10 题完全脱稿

如果面试时间有限，我建议你优先把下面十个练到我随便追问都能接住：

1. **为什么 RRF 不够？**
2. **为什么 Listwise？**
3. **Gold relevance 怎么来的？**
4. **NDCG@20 怎么算？为什么适合这个项目？**
5. **为什么两阶段 SFT？**
6. **SFT 明明只是 CE，怎么学 ranking？**
7. **为什么 SFT 后还需要 RL？**
8. **为什么 GSPO 而不是 GRPO/PPO/DPO？**
9. **GSPO 的 advantage、ratio、clip 分别怎么算？**
10. **Reward 到底怎么构造，怎么防 reward hacking？**

这 10 个实际上就构成了一条完整的面试链：

RRF→Listwise→Gold→NDCG→SFT→ObjectiveMismatch→GSPO→Reward

你真正要达到的状态不是“背完 60 道题”，而是**无论面试官从其中哪个节点插进去，你都能够顺着因果链往前解释为什么、往后解释怎么实现。**