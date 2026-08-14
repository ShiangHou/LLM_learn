# 基于OPD的基金推荐论据生成
## 背景

![img](https://cdn.jsdelivr.net/gh/ShiangHou/markdown_image/imglink.png)

### 输入

输入是KYC，KYP，KYCP，KYM，KYPM数据

> 分别叫做用户数据，产品数据，市场数据

以及prompt

```josn
{"基金代码": "519981", "基金简称": "长信标普100等权重指数（QDII）人民币份额", "基金全称": "长信美国标准普尔100等权重指数增强型证券投资基金", "成立时间": "2011-03-30 00:00:00", "基金类型代码": 6, "基金类型分类名称": "权益型", "一级分类名称": "QDII", "二级分类名称": "QDII股票型", "三级分类名称": "QDII股票型", "风险等级代码": "4", "风险等级名称": "中高", "基金状态代码": "0", "上架标识：1上架0下架": "1", "定投状态代码": "0", "定投最小金额": "1.0", "赎回到账天数": "5", "申购最小金额": "1.0", "日累计申购限额": "1000000.000000", "管理费率": "1.100000", "托管费率": "0.300000", "权益类投资占资产总值比例": "0.801600", "投资目标描述": "    本基金通过增强型指数化的投资来追求美国超大市值蓝筹股股票市场的中长期资本增值，通过严格的投资纪律约束和数量化风险管理手段力争将基金净值增长率和标普100等权重指数（本基金的标的指数）收益率之间的日均跟踪误差控制在0.50%以内（相应的年化跟踪误差控制在7.94%以内）。在控制跟踪误差的前提下通过有效的资产及行业组合和基本面研究，力求实现对标的指数的有效跟踪并取得优于标的指数的收益率。", "投资策略描述": "    本基金为股票指数增强型基金，原则上不少于80%的基金净资产采用完全复制法，按照成份股在标普100等权重指数中的基准权重构建指数化投资组合，并根据标的指数成分股及其权重的变化、进行相应的调整。当预期成份股发生调整和成份股发生配股、增发、分红等行为时，或因基金的申购和赎回等对本基金跟踪标的指数的效果可能带来影响时，或因某些特殊情况导致流动性不足时，或其他原因导致无法有效复制和跟踪标的指数时，基金管理人可以对投资组合管理进行适当变通和调整，并辅之以金融衍生品投资管理等，力争使跟踪误差控制在限定的范围之内。\\r\\n    对于不高于基金净资产15%的主动型投资部分，基金管理人会根据深入的公司基本面和数量化研究以及对美国及全球宏观经济的理解和判断，在美国公开发行或上市交易的证券中有选择地重点投资于基金管理人认为最具有中长期成长价值的企业或本基金允许投资的固定收益类品种、货币市场工具、金融衍生品以及有关法律法规和中国证监会允许本基金投资的其他金融工具。", "投资品种比例限制": "    本基金主要投资于标普100等权重指数成份股，同时也可主动投资于下列金融产品或工具：权益类品种（包括在美国证券市场挂牌交易的普通股、优先股、美国存托凭证、交易型开放式指数基金等）；固定收益类品种（包括美国政府债券、美国市场公开发行的公司债券及可转换债券等债券资产）；回购协议、美国或中国短期政府债券、现金等价物、货币市场基金等货币市场工具；经中国证监会认可的境外交易所（美国）挂牌交易的金融衍生品（包括期权期货）；以及法律、法规或中国证监会允许基金投资的其他金融工具。\\r\\n    本基金为股票指数增强型基金，对标普100等权重指数的成份股的投资比例为基金资产净值的80%-95%；现金（不包括结算备付金、存出保证金、应收申购款等）、现金等价物及到期日在一年以内的美国或中国短期政府债券的比例为基金资产净值的5%-20%；主动投资部分占基金资产净值的0%-15%，投资于在美国证券市场公开发行或挂牌交易的权益类品种、固定收益类品种、金融衍生品以及有关法律法规和中国证监会允许本基金投资的其他金融工具。\\r\\n    如法律法规或监管机构以后允许基金投资其他品种，本基金管理人在履行适当程序后，可以将其纳入投资范围。", "成立以来回报率": "156.1467", "近1年年化收益率排名": "329/661", "近1年超额收益": "0.0146176876", "近1年超额收益同类排名": "1/2", "近1年最大回撤": "0.1138475837", "近1年最大回撤同类排名": "1/2", "近1年最大回撤修复天数": "21.0", "近1年年化波动率": "0.1408059836", "近1年年化波动率同类排名": "1/2", "近1年夏普率": "0.3785983914", "近1年夏普率同类排名": "1/2", "近1年卡玛比率": "0.5265344786", "近1年卡玛比率同类排名": "1/2", "近1年跟踪误差": "2.9759", "第一基金经理": "傅瑶纯", "第二基金经理": "", "管理的基金经理人数": "1", "基金规模": "8.3626537601E8", "当前持有中用户数": 645, "近30天持有用户数": 694, "当前持仓金额": 10859395.889999993, "近7天AUM增长金额": 617441.0599999912, "热度综合得分": "44", "近30天浏览次数": "1800", "近30天搜索次数": 2320, "近30天搜索人数": 1324, "近30天加自选人数": 255, "近30天申购次数": "3685", "近30天申购用户数": "2374", "近一周复购率": "0.84861976", "最新报告期申赎增量": "1.72496736E8", "最新报告期申赎增速": "86.73368"}
```

### 输出

一段10个字以内的标题+30个字的标签



## 目标

当前的推品没有与论据相匹配（资讯内容，工具、产品标签、用户客群），缺少统一的财富论据平台，差异化论据推荐成本高，需要针对不同用户和不同产品生成更有针对性的推荐论据，



最后需要在基金产品页/搜索页上展示

## 评测标准

《基金论据心理型运营评判标准》在多个维度上打分，和排序，给出ABCD四个档位





## 模型选型

![img](https://cdn.jsdelivr.net/gh/ShiangHou/markdown_image/imglink-20260801201125633.png)

先进行测评，测试包括claude上的一些模型，最后觉得deepseek v4 flash效果最好，然后是glm，qwen3 235B-A22B等，但其实A+B率都在75%，A率都在0.5%，



最终目标是训出来一个A率能够很高的，



## 训练流程

思路是先进行teacher模型的选择，准备数据，然后学生模型的选择，蒸馏，检验问题，因为A率最高的是deepseek v4 pro所以先进行数据生成

公募基金数据 → teacher模型蒸馏 → 数据清洗 → 数据增强（规则、MinHash语义去重、Self-Consistency、LLM-as-Judge）→ 基于《基金论据心理型运营评判标准v1》做多维质量评测打分和排序 → 筛选top1w高质量数据构造大模型训练数据 → 边际case DPO偏好训练

筛选出了1w条A的数据，



初版A+B率能到61%，然后A率能到3.2%，



然后分析数据，得A的大多数是同一种组合，然后又去搞数据，运营给出了按照基金类型（即按照产品KYC分层构建思维连），相当于固定了思维链条，针对边界长尾的数据进行了DPO（即给ABCD，选择A这样）



所以经过sft后，得到了一个质量还可以的30B的模型，



但是出现了同质化，即高质量的数据出现在了相同的证据组合上，对于一些其他的证据组合，效果并不好，原因在于运营给的固化了一条思维链，对于这些组合是ok的，对于其他的组合就不ok



此外，组合过于固定，无法获取全部组合的思维链，即无法得到



但是风格已经学好了，主要问题在于组合-思维链的探索上



# 面试说法
## 简单介绍一下你的项目

好的面试官，首先还是先说一下背景。我们这个场景是在推荐品的时候，需要结合用户+市场+产品特征，给出一段标题10个字，正文20个字的一个推荐理由，因此叫做“论据”，这个论据最后会挂在搜索结果页和用户的持仓页上。

论据模型的输入是三大维度的特征，包括用户信息（风险偏好），市场信息（主要是指数，环境之类的），产品信息（基金产品信息，比如收益率之类的）三个加一起大概有50个指标。讲这些特征输入后，模型需要输出一段10个字，正文20个字的一个路由

在定义什么是“好论据”的时候，运营同学给了一个“人类心理学论据标准“，将论据结果分为了ABCD四个档次，我们在用大模型测试的时候，绝大多数的模型的A+B率都只有0.5%左右，然后我们通过过采样，去重，、Self-Consistency，LLMjudge ，按照运营的标准筛选了1W条高质量数据做sft

SFT 后模型已经能稳定生成合规、流畅的论据，但 A 档比例仍然有限，所以我们又针对 A/B、B/C 这种边界 case 构造 preference pair 做 DPO，让模型学习运营质量边界。

后来进一步分析发现，模型虽然语言质量提高了，但高质量输出严重集中在少数几个指标组合上，本质不是“怎么写”的问题，而是 P(z∣x)，也就是 evidence selection 发生了模式集中。

所以模型实际上要同时完成两个任务：第一是 Evidence Selection，也就是选择哪些指标值得讲；第二是 Generation，也就是怎么把这些指标组织成符合运营要求的论据。

所以后续我们把高质量证据组合作为 特权信息，引入 OPSD。同一个模型承担 student 和 teacher 两种角色，student 只看到原始基金特征并进行 on-policy rollout，而 teacher 额外看到正确的 evidence plan，在 student 自己访问的 token state 上提供 dense distribution supervision，从而把“选择哪些证据”和“如何组织论据”一起蒸馏到最终单模型里。

最终离线 A+B 达到 91%，A 达到 10.2%，同时基于 evidence combination 的多样性指标提升约 20%。

# 第一层：项目介绍

这是“HR/业务面 + 技术面开场”层面。要求你 **1～2 分钟无停顿讲完**。

## Q1：先介绍一下这个项目

### 标准答案

> 这个项目是面向基金产品页和搜索推荐场景，根据用户画像、基金产品属性以及市场行情等多源特征，自动生成一个短标题和大约 30 字左右的基金推荐论据。
>
> 这个任务表面上是文本生成，但真正的难点是输入特征很多，比如收益率、最大回撤、夏普率、基金规模、跟踪指数、搜索热度、持仓增长等，而最终论据只能使用少量信息，所以模型实际上要同时完成两个任务：第一是 **Evidence Selection，也就是选择哪些指标值得讲**；第二是 **Generation，也就是怎么把这些指标组织成符合运营要求的论据**。
>
> 我们第一阶段先横评多个大模型作为 Teacher，然后通过多采样、规则过滤、MinHash 去重、Self-Consistency 和 LLM-as-Judge，按照运营制定的 A/B/C/D 标准筛出约 1 万条高质量数据，对 Qwen3-30B-A3B 做 SFT。
>
> SFT 后模型已经基本学会了输出格式和写作风格，但是对 A/B 这种质量边界学习得不够，所以进一步针对边界样本做 DPO。
>
> 后来我们分析发现，即使整体质量提升，高质量论据还是集中在少数固定的指标组合上，本质不是“不会写”，而是证据选择发生了模式集中。所以最后通过 OPSD，把高质量的 evidence plan 作为训练阶段的 privileged information，让模型在 on-policy trajectory 上同时学习“选哪些指标”和“怎么组织论据”。
>
> 最终离线评测 A+B 达到 91%，A 达到 10.2%，指标组合多样性提升约 20%。

### 面试官在听什么？

不是听你用了：

> SFT + DPO + OPD。

而是在听：

业务问题→训练→发现失败模式→分析原因→针对性解决

你的项目最有竞争力的是这个闭环。

------

# 第二层：业务和数据

------

## Q2：为什么这个问题需要大模型？规则不行吗？

### 答案

规则比较适合：

feature→fixed template

比如：

> 如果近一年收益率 > X，就输出“近期业绩表现较好”。

但真实基金有几十甚至上百个候选特征，而且不同基金类型适合讲的东西不同。

例如：

QDII 指数基金可能关注：

跟踪指数+超额收益

主动权益基金可能关注：

基金经理+回撤+收益

债基可能更关注：

波动率+最大回撤+收益稳定性

所以真正的问题是：

P(z∣x)

也就是：

> 给定当前基金状态，应该从大量特征中选哪几个。

这很难靠有限规则枚举。

------

# Q3：你的输入具体有什么？

不要回答：

> KYC、KYP、KYM。

这太虚。

你应该举例。

### 产品侧

包括：

- 基金类型；
- 风险等级；
- 跟踪指数；
- 投资策略；
- 成立时间；
- 基金规模；
- 收益率；
- 超额收益；
- 同类排名；
- 最大回撤；
- 波动率；
- 夏普率；
- 卡玛比率；
- 跟踪误差。

### 行为/热度侧

比如：

- 持有用户数；
- 近 30 天浏览；
- 搜索；
- 加自选；
- 申购；
- 复购率；
- AUM 增长。

这些字段在你的实际输入样例里都有。

------

# Q4：输出是什么？

### 答案

两个字段：

y=(title,argument)

例如：

> 标题：美股核心布局
>  论据：跟踪标普100指数，近一年超额收益表现居前。

其中标题控制在 10 字以内，论据大约几十个字，最终用于产品页/搜索页展示。你的原始文档就是这样定义输出和应用场景的。

------

# Q5：这个任务最难的地方是什么？

这是非常关键的问题。

### 推荐答案

> 最开始我们认为难点是生成质量，但做完 SFT 和 DPO 以后发现，更核心的问题其实是 evidence selection。
>
> 因为输入里有大量候选信息，但最终只能讲 2～3 个，因此可以把任务拆成：
>
> xEvidence SelectorzGeneratory
>
> 后期我们发现 P(y∣x,z) 已经学得比较好，真正的问题集中在 P(z∣x)。

这句话：

P(y∣x,z)没大问题，主要瓶颈是 P(z∣x)

最好背下来。

------

# 第三层：Teacher 和数据构造

------

# Q6：为什么需要 Teacher 造数据？

### 答案

因为真实线上很难直接拿到大量：

(x,高质量论据)

监督数据。

运营人工写 1 万条成本很高。

因此使用强模型：

xTeachery

生成候选，再通过 Judge 和规则进行 rejection sampling。

------

# Q7：Teacher 怎么选？

文档里记录了 Claude、DeepSeek、GLM、Qwen 等模型的横评，并且整体 A+B 在约 75%，但是 A 只有约 0.5%。

### 回答方式

> 我们没有只看模型规模，而是在统一 validation set、统一 prompt 和统一运营 rubric 下比较不同 Teacher。
>
> 对业务而言不仅看 A+B，更重要的是最高质量 A 的比例，所以最终选择最高质量档更稳定的模型做主要数据生成 Teacher。

------

# Q8：Teacher A率才0.5%，为什么最后学生能做到10.2%？

这是**高频压力题**。

### 答案

因为 Student 并不是拟合 Teacher 的原始分布：

pT(y∣x)

我们进行了多采样和筛选。

对于一个 x，可以采样：

y1,…,yK∼pT(y∣x)

Judge：

si=J(x,yi)

然后：

y∗=argimaxsi

所以训练集实际上更接近：

pT(y∣x,q(y)≥τ)

即：

> Teacher **高质量条件分布**。

因此 Student 完全可能比 Teacher 单次 zero-shot sampling 的 A 率高。

一句话版本：

> **Teacher 的单次生成质量不是 Student 的上界，我们实际上做了 rejection sampling。**

------

# Q9：为什么一个 input 要采样多个答案？

因为单次：

y∼p(y∣x)

方差很大。

通过 Best-of-N：

y∗=argyimaxJ(x,yi)

可以获得 Teacher 分布里的优质长尾样本。

------

# Q10：Self-Consistency 在这里是什么？

不要把 Self-Consistency 只理解成数学题投票。

这里可以解释成：

> 对同一个 input 通过不同 sampling trajectory 获得多条候选，然后判断核心证据是否稳定、内容是否一致，同时配合 Judge 做候选筛选。

本质是：

x→{y1,…,yK}

而不是：

x→y

一次决定。

------

# Q11：为什么要 MinHash？

因为 Teacher 大量生成之后会出现很多 near-duplicate：

> 近期业绩表现突出
>  近期业绩表现亮眼
>  近期表现优异

如果这种样本占大量训练数据：

Pdata(某种模式)↑

SFT 会进一步强化这种模式。

MinHash 用来高效近似计算集合的：

Jaccard(A,B)=∣A∪B∣∣A∩B∣

文本一般先转成 n-gram / shingle 集合。

------

# Q12：MinHash 和 embedding 去重有什么区别？

### MinHash

比较的是：

Jaccard Similarity

偏 lexical overlap。

### Embedding

比较：

cos(e1,e2)

偏 semantic similarity。

所以：

> MinHash 擅长大量数据的近重复清洗；Embedding 更适合语义同义但表面文字差别比较大的情况。

------

# 第四层：LLM-as-Judge

------

# Q13：Judge 为什么不能直接让模型输出 A/B/C/D？

可以，但不好 debug。

更好的设计是：

J(x,y)→(s1,…,sk,grade,reason)

例如：

```
{
  "grounded": true,
  "evidence_quality": 4,
  "logic": 4,
  "differentiation": 3,
  "expression": 4,
  "grade": "A"
}
```

然后根据运营 rubric 映射到 A/B/C/D。

你的文档能够确认的是“运营从多个维度进行评分并划分 ABCD”；具体维度和权重文档没有记录，所以面试不能凭空编权重。

------

# Q14：你怎么证明 LLM Judge 靠谱？

要有人类 Gold Set。

比如运营人工标：

Dhuman=500

然后比较：

HumanvsLLM Judge

最简单：

Accuracy=N#correct

但 ABCD 是有序标签，所以还可以看：

Weighted Cohen′s Kappa

------

# Q15：Judge最重要看 Precision 还是 Recall？

如果 Judge 主要用来筛训练数据：

> **高质量档 Precision 更重要。**

比如 A：

PrecisionA=Judge=AHuman=A∩Judge=A

因为 False Positive 会直接污染 SFT 数据。

宁愿少拿一些 A：

RecallA↓

也不希望大量 C 被判成 A。

------

# Q16：LLM-as-Judge 有什么 bias？

必须知道：

- position bias；
- verbosity bias；
- self-preference/self-enhancement；
- prompt sensitivity；
- style bias。

因此不能认为：

LLM Judge=Ground Truth

它必须经过人工 calibration。

------

# 第五层：SFT

------

# Q17：为什么第一阶段用 SFT？

因为第一阶段需要让模型快速学会：

- 任务格式；
- 输出长度；
- 语言风格；
- grounding；
- 哪类特征通常值得使用。

SFT 是最高效的 imitation learning。

------

# Q18：SFT loss 是什么？

必须秒写。

给定：

x

输出：

y=(y1,…,yT)

模型：

P(y∣x)=t∏P(yt∣x,y<t)

Loss：

LSFT=−T1t∑logPθ(yt∣x,y<t)

本质：

> 每个 token 做一次 vocab size 的多分类 Cross Entropy。

------

# Q19：Prompt 部分也算 loss 吗？

通常 instruction tuning：

prompt token：

mask=−100

只在 response token 上计算 loss。

否则模型会花能力学习：

> 复述用户输入。

------

# Q20：为什么 SFT 之后还需要 DPO？

SFT 只知道：

yA是一个正确答案

它不知道：

yA>yB

例如 A 和 B 都是合法论据。

SFT 对二者都是：

−logp(y∣x)

而运营真正的信息是：

A>B>C>D

所以需要 preference learning。

------

# 第六层：DPO

------

# Q21：DPO数据怎么构建？

同一个 x：

(x,yw,yl)

例如：

(A,B)

或者：

(B,C)

重点应该选择：

A↔B

这种 boundary case。

------

# Q22：为什么不是 A vs D？

因为：

A≫D

通常模型本来就知道。

这种 pair：

gradient information

很有限。

真正有价值的是：

> 为什么一个已经不错的 B 仍然没有达到 A。

所以：

A≳B

信息密度更高。

------

# Q23：DPO loss？

必须会。

LDPO=−logσ[β(logπref(yw∣x)πθ(yw∣x)−logπref(yl∣x)πθ(yl∣x))]

------

# Q24：reference model 是谁？

通常：

> DPO 开始之前冻结的 SFT checkpoint。

πref=πSFT

------

# Q25：为什么需要 reference？

因为如果没有 reference，只要求：

P(yw)>P(yl)

策略可能无限漂移。

DPO 实际学习的是：

Δw=logπθ(yw)−logπref(yw)

与：

Δl

之间的差异。

------

# Q26：β 是什么？

可以近似理解成：

> preference optimization strength 和离 reference model 偏离程度之间的控制参数。

不要简单回答：

> “β越大更新越大。”

更准确是它改变 preference margin 的尺度。

------

# 第七层：项目真正的灵魂——同质化

------

# Q27：你怎么发现模型同质化？

SFT/DPO 后分析 A 类样本，发现：

zi={selected evidence}

高度集中在少数几个组合。

例如很多答案都使用：

{收益率,同类排名}

即使输入其实还有很多高质量证据。

你的文档明确记录了：

> 高质量输出集中在相同证据组合，固定思维链能够提高这些组合的质量，却无法覆盖所有组合。

------

# Q28：这为什么不只是语言表达同质化？

非常关键。

表达同质化是：

P(surface form∣z)

比如：

> “表现突出”
>  “表现亮眼”

证据同质化是：

P(z∣x)

模型永远选：

收益+排名

你的项目遇到的主要是第二种。

------

# Q29：你怎么证明真正的问题在 Evidence Selection？

进行 error analysis。

把最终输出拆成：

x→z→y

然后分别检查：

### 当给定正确 z 时

模型能不能写好？

如果能：

P(y∣x,z)

没问题。

### 不提供 z

模型选的 z 是否集中？

如果是：

P(z∣x)

出了问题。

所以最终判断：

> surface realization 已经比较成熟，planner/evidence selection 成为瓶颈。

------

# Q30：为什么继续加 SFT 数据不行？

如果新增数据仍然是：

z1,z1,z1,z1,z2

那么：

N↑

但是：

Support(z)

没有明显增加。

甚至：

P(z1)↑

模式塌缩更加严重。

一句很好的回答：

> **我们的瓶颈已经从 data quantity 转变成了 support coverage。**

------

# 第八层：OPD

这里开始是“大模型后训练岗真正区分度”的部分。

------

# Q31：OPD 是什么？

On-Policy Distillation。

传统 KD/SFT 通常在：

Teacher trajectory

上训练。

OPD：

y∼πstudent

让 Student 自己 rollout。

然后在 Student 真正访问的 prefix：

(x,y<t)

让 Teacher 提供：

PT(⋅∣x,y<t)

Student 对齐 Teacher。

------

# Q32：为什么叫 On-Policy？

因为：

trajectory∼πθ

来自**当前 student policy**。

和 Teacher 是否在线更新没有关系。

------

# Q33：OPD 和 SFT 最核心区别？

### SFT

state：

(x,y<t∗)

其中 y∗ 是专家答案。

supervision：

one−hot(yt∗)

------

### OPD

state：

(x,y^<t),y^∼πS

supervision：

PT(⋅∣x,y^<t)

所以：

SFT=expert state+hard labelOPD=student state+soft label

------

# Q34：OPD解决什么问题？

主要解决：

dtrain(s)=dtest(s)

也就是 exposure/distribution mismatch。

SFT 从来只教模型：

> 在正确历史下下一步怎么办。

OPD 可以教模型：

> 你自己已经生成成这样了，现在下一步应该怎么办。

------

# Q35：OPD loss？

Student rollout：

y^∼πS

在每个 t：

pSt=πS(⋅∣x,y^<t)pTt=πT(⋅∣x,y^<t)

例如 Reverse KL：

L=T1t∑KL(pSt∣∣pTt)

------

# 第九层：OPSD

------

# Q36：OPSD 和 OPD 有什么区别？

普通 OPD：

External Teacher→Student

你的 OPSD：

> 同一个基础模型承担 student role 和 teacher role，但是看到的 context 不一样。

Student：

Pθ(y∣x)

Teacher role：

Pθ(y∣x,z∗)

其中：

z∗

是额外 privileged evidence。

------

# Q37：同一个模型怎么自己蒸馏自己？不是完全一样吗？

不是。

虽然：

θT=θS

但是：

ContextT=ContextS

Student 只知道：

x

Teacher 知道：

(x,z∗)

于是：

P(y∣x,z∗)=P(y∣x)

差异来自**information asymmetry**。

------

# Q38：你项目里的 privileged information 到底是什么？

这是一定要答死的。

> **经过筛选的高质量 evidence plan，也就是当前基金最适合用于论据生成的指标组合。**

比如：

```
原始特征：
收益、回撤、夏普率、基金规模、搜索量、
AUM、跟踪指数、超额收益……

privileged evidence：
[跟踪指数，超额收益排名]
```

------

# Q39：OPSD具体怎么训练？

第一步：

Student rollout：

y^∼πθ(⋅∣x)

第二步，在相同 Student prefix：

y^<t

计算 student：

pS=πθ(⋅∣x,y^<t)

第三步 teacher role 多看 z∗：

pT=πθ(⋅∣x,z∗,y^<t)

最后：

LOPSD=T1t∑KL(pSt∣∣stopgrad(pTt))

------

# Q40：为什么一定要 stop-gradient？

如果 teacher 和 student 两侧一起优化：

pS→pT

同时：

pT→pS

target 本身也在动。

监督信号会不稳定甚至退化。

所以 teacher role：

stopgrad(pT)

作为 target。

------

# Q41：既然 teacher 知道 evidence，为什么不直接 SFT teacher 的最终答案？

这是非常强的追问。

### 回答

因为那又回到了：

expert trajectory

而 student 在推理时会访问自己的：

student trajectory

OPSD 的价值在于：

y^∼πS

然后 Teacher 在 Student 自己已经走到的状态上继续指导它。

所以不仅是：

> “学习好答案”。

更重要：

> “在自己的错误分布上接受 dense correction。”

------

# 第十层：最容易把项目问穿的问题

------

# Q42：你没有显式预测 evidence token，凭什么说学到了 evidence selection？

这是你最应该练的问题之一。

### 推荐答案

> OPSD 并不是直接监督一个 evidence classification head，而是用 evidence-conditioned teacher distribution 对最终生成 token 做蒸馏。
>
> 给定不同的 z，Teacher 的 token distribution 会显著不同，比如看到“超额收益排名”以后，“超额、排名、同类”等 token probability 会提升；看到“跟踪指数”以后，相应指数和资产主题 token probability 会提升。
>
> 所以大量不同 x,z 条件下进行 distribution alignment，本质是在隐式蒸馏：
>
> x→z→y
>
> 的能力。
>
> 但是严格来讲，它不是显式 evidence selector，因此我们额外通过 evidence extraction 和组合统计去验证模型最终确实扩大了证据组合 support，而不是只看生成文本 diversity。

最后一句很加分。

------

# Q43：OPSD为什么能提高 diversity？KL不是容易模式集中吗？

**绝对不能回答：**

> OPD天然提高多样性。

错误。

### 正确答案

> Diversity 不是 OPSD 自己创造出来的。
>
> 真正的来源是 privileged evidence distribution：
>
> P(z∣x)
>
> 本身覆盖了更多高质量组合。
>
> OPSD 负责的是把这些 diverse evidence plans 蒸馏给 student。

所以：

Diverse Evidence+OPSD→Diverse Student

而不是：

OPSD→Diversity

------

# Q44：如果 evidence plan 本身也很单一呢？

那 OPSD 解决不了。

Pprivileged(z)

如果已经塌缩：

P(z1)=90%

Student 最后依然会学成：

z1

所以：

> 上游 evidence generation 的 coverage 是整个方案成立的前提。

------

# Q45：为什么不把 Planner 和 Generator 都上线？

因为两阶段：

Planner→Generator

会带来：

- 两次推理；
- latency；
- GPU成本；
- 两模型维护；
- planner error propagation。

而 OPSD 的目标是：

训练阶段：

Planner/evidence→privileged teacher

线上：

x→single model→y

------

# Q46：那为什么不直接让最终模型输出 evidence + answer？

可以，这是 alternative baseline。

例如：

```
Evidence:
1. 跟踪指数
2. 超额收益

Answer:
……
```

优点：

- 可解释；
- 显式 supervision。

缺点：

- 在线多生成 token；
- 用户不需要看到 reasoning；
- 模型可能 evidence 写对但 answer 不严格遵循；
- 输出协议更复杂。

因此 OPSD 的目标是：

> **训练时显式提供能力，推理时隐式内化能力。**

------

# 第十一层：指标拷打

------

# Q47：A率怎么算？

A Rate=NNA

------

# Q48：A+B怎么算？

AB Rate=NNA+NB

------

# Q49：为什么两个都报？

因为：

A+B

反映 overall usability。

A

反映 top-quality capability。

你的结果：

A+B=91%

代表整体高质量输出比较稳定。

A=10.2%

代表真正精品的比例。

------

# Q50：你的“多样性提升20%”怎么算？

这是你必须统一口径的地方。

推荐使用 Evidence Combination Entropy。

每条结果提取：

ci={fi1,fi2,...}

第 k 类组合频率：

pk=Nnk

entropy：

H=−k∑pklogpk

归一化：

Hnorm=logKH

然后：

Improvement=HbeforeHafter−Hbefore

比如：

0.50→0.60

则：

+20%

注意：**你的原始项目文档没有记录 20% 的原始计算方法。**所以你应该回实验代码/报表确认真实口径；如果这个 20% 是后来为了简历汇总的指标，就建议统一成上述 entropy 定义，而不要面试现场临时编。

------

# Q51：只看 entropy 有什么问题？

模型可以胡乱选择各种证据：

Diversity↑

但：

Quality↓

所以必须联合看：

(A+B, Hnorm)

理想情况：

A+B↑

同时：

Hnorm↑

叫：

> **Quality-Diversity Tradeoff。**

------

# 第十二层：KL 八股追问

------

# Q52：Forward KL？

Teacher q，Student p：

KL(q∣∣p)=x∑q(x)logp(x)q(x)

如果：

q(x)很大,p(x)很小

惩罚大。

所以倾向：

> mode-covering。

------

# Q53：Reverse KL？

KL(p∣∣q)=x∑p(x)logq(x)p(x)

如果：

p(x)很大,q(x)很小

惩罚大。

所以 Student 不敢进入 Teacher 不认可区域。

倾向：

> mode-seeking。

------

# Q54：你的项目本来缺 diversity，为什么用 reverse KL？

回答：

> Diversity 的 source 不是 KL direction，而是不同 privileged evidence 对应的 target modes。
>
> Reverse KL 主要负责保证 Student 在每一个 evidence-conditioned local distribution 上对齐 Teacher，而 evidence coverage 负责扩大整体 mode support。

------

# Q55：为什么 soft logits 比 Teacher 最终答案信息更多？

hard label：

[0,0,1,0,...]

只知道：

> 正确 token 是哪个。

Teacher distribution：

[0.01,0.10,0.65,0.20,...]

还告诉 Student：

> alternative token 之间的相对合理程度。

这就是 distillation 中经常讲的：

> dark knowledge。

------

# Q56：只有 Teacher top-5 logprob 能做完整 OPD 吗？

不能精确计算：

KL(p∣∣q)=v∈V∑p(v)logq(v)p(v)

因为需要整个词表：

V

的 teacher probability。

top-k 只能做：

- truncated KL；
- top-k renormalized KL；
- sampled approximation。

------

# 第十三层：为什么不是其他算法

------

# Q57：为什么不用 PPO？

因为：

1. 已经有 ABCD preference；
2. PPO 需要额外 reward model / critic 等训练链路；
3. 训练成本和稳定性复杂很多；
4. 业务 reward 很难天然 verify。

所以 DPO 更直接。

------

# Q58：为什么不用 GRPO？

GRPO 更适合：

reward(y)

能比较稳定评估的任务。

比如数学：

answer correct=1

你的业务 reward 本质是：

LLM Judge

如果直接 RL：

θmaxE[J(x,y)]

容易学 Judge shortcut / reward hacking。

而 OPSD 给的是：

token−level dense supervision

对 evidence selection 更直接。

------

# Q59：为什么不是纯 DPO 一路训到底？

因为 DPO 主要解决：

P(yw∣x)>P(yl∣x)

的 preference boundary。

但你的 failure mode 已经变成：

P(z∣x)

支持集太窄。

如果 preference dataset 里的 evidence coverage 本身不足，DPO 也会继续强化已有 mode。

------

# 第十四层：Qwen3-30B-A3B 和 MoE

------

# Q60：为什么选30B-A3B？

三点。

第一：

> 输入特征复杂，任务实际上需要进行 evidence selection，而不是因为答案长。

第二：

> 30B 总 capacity 足够。

第三：

> MoE 每 token 只激活部分参数，可以获得较大总容量，而实际 active parameters 更低。

------

# Q61：30B-A3B是什么意思？

大体含义：

Total Params≈30B

但单 token 激活：

Active Params≈3B

所以：

Total Capacity=Active Compute

------

# Q62：MoE由什么组成？

普通 Transformer FFN：

x→FFN(x)

MoE：

xRouterExperti

router 给：

p(e∣x)

通常选 top-k experts。

输出：

y=e∈TopK∑p(e∣x)Ee(x)

------

# Q63：MoE最大的训练问题？

Expert imbalance。

某些 expert：

load≫others

导致：

- token drop；
- GPU负载不均；
- 部分 expert 学不到东西。

因此通常存在 load balancing mechanism/loss。

------

# 第十五层：训练工程追问

------

# Q64：为什么训练集不能随机切？

因为同一基金不同时间的数据非常相似。

如果：

```
fund_code=519981
```

同时进入 train/test：

模型可能已经见过：

- 基金名称；
- 类型；
- 跟踪指数；
- 投资策略。

导致 leakage。

更严格应该做：

GroupSplit(fund_code)

或者：

TimeSplit

最好：

Entity+Time

------

# Q65：怎么防止 hallucination？

你的场景属于 grounding generation。

核心原则：

Generated claim⊆Input evidence

可以：

- prompt 强约束；
- SFT grounding；
- rule validator；
- Judge factuality；
- 数值字段 exact match；
- unsupported claim 检测。

尤其金额、收益率、排名不能凭空修改。

------

# Q66：如果输入中有错误数据怎么办？

模型解决不了 source truth 问题。

要区分：

Data Error

和：

Generation Error

如果 source 本身错误，模型 faithfully grounded 也会输出错误结果。

所以系统需要：

> 上游数据治理 + 下游 generation grounding。

------

# 第十六层：终极压力题

这些问题最适合你真正去练。

------

## Q67：你这个项目听起来不就是“Teacher造数据 + SFT + DPO”，OPSD是不是为了简历硬加的？

### 高质量回答

> 如果只是提升文字流畅度，确实 SFT+DPO 已经足够，没有必要加入 OPSD。
>
> 我们加入它的原因来自明确的 error analysis：SFT/DPO 后高质量结果并不是随机失败，而是集中在固定 evidence combination；在手工给定高质量 evidence 的情况下，Generator 能够很好完成任务。因此 bottleneck 很明确从 Generation 转移到了 Evidence Selection。
>
> 所以后续训练目标也相应从“优化整体文本 preference”变成“把 Evidence Selection 能力蒸馏给最终单模型”。OPSD 是针对这个 failure mode 引入的，而不是先选算法再找场景。

这题答好非常加分。

------

# Q68：为什么不把 evidence selection 做成分类模型？

### 答案

可以，而且是很合理的 baseline。

但它的问题是：

候选 evidence 是：

2N

组合空间。

不是简单单标签分类。

而且不同 feature 有 interaction。

例如：

收益高

单独不一定值得讲。

但：

收益高+同类排名高

组合才有意义。

所以它更接近：

structured selection

而不是普通 multi-class classification。

------

# Q69：为什么你最后A只有10.2%，听着很低？

非常现实的问题。

回答：

> A 是运营最高标准，并不等于“合格”。我们的 production usability 主要通过 A+B 衡量，达到 91%；A 代表非常强的差异化精品文案，所以它本身是严格的长尾指标。
>
> 相比 Teacher zero-shot 约 0.5% 的 A rate，最终 10.2% 已经是一个数量级的提升。

------

# Q70：为什么A+B高，A这么低？

因为：

B→A

不是单纯减少错误。

A 往往要求：

- evidence 更优；
- 逻辑更强；
- 差异化更好；
- 文案更精炼。

所以这是：

acceptable→exceptional

的提升，比：

wrong→acceptable

更难。

------

# Q71：你觉得这个项目最大的问题/遗憾是什么？

别说：

> 时间不够。

可以回答：

> 最大的问题是离线 Judge 仍然不能完全替代真实用户反馈。目前训练和评测主要围绕运营质量标准，最终真正目标其实是 CTR、点击后转化以及用户长期体验。
>
> 所以后续更理想的是做：
>
> Offline Quality+Online Business Metrics
>
> 的联合评估，并检查 diversity 提升是否真正改善用户体验，而不是只改善离线文本指标。

这是一个非常成熟的答案。

------

# 最后：你现在应该达到什么程度

这个项目你应该练到以下几个问题能**完全脱稿**：

1. 2 分钟介绍完整项目。
2. 为什么任务核心是 Evidence Selection。
3. Teacher A 0.5%，Student 为什么能 10.2%。
4. SFT loss。
5. 为什么 SFT 后做 DPO。
6. DPO loss。
7. 为什么选择 boundary case。
8. 同质化到底怎么发现。
9. 为什么不是简单增加 SFT 数据。
10. OPD 和 SFT 区别。
11. 什么叫 on-policy。
12. OPD KL 怎么计算。
13. OPSD 和普通 OPD 区别。
14. privileged information 是什么。
15. 为什么同一个模型可以自己教自己。
16. 为什么 OPSD 能隐式学习 evidence selection。
17. **为什么不能说“OPSD天然提高多样性”。**
18. diversity 20% 怎么计算。
19. 为什么不用 GRPO。
20. 为什么最终只上线一个模型。

其中我认为真正决定这个项目能不能拿高评价的是 **第 7～17 题**。前面 SFT/DPO 很多候选人都会；但如果面试官问到：

> “你的问题究竟发生在 P(z∣x) 还是 P(y∣x,z)？”
>  “OPSD 又没有监督 evidence token，凭什么说它蒸馏了 evidence selection？”
>  “Reverse KL 本身 mode-seeking，你为什么还能提高 diversity？”

你还能把这三个问题讲明白，那这个项目就会从“做过后训练”升级成**真正理解自己在解决什么模型问题**。
