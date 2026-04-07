# Qwen3 系列模型框架

报告汉语版翻译https://zhuanlan.zhihu.com/p/1905945819108079268
## Qwen3 种类

qwen3 的dense有 0.6B，1.7B 4B 8B 14B 32B

Moe框架有Qwen3-30B-A3B，Qwen3-235B-A22B

值得注意的是，论文里面说

For smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge transfer from larger models to enhance their capabilities. Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency.

就是说小模型是直接从训好的那个大模型蒸馏出来的，

dense模型用的架构分别是
（GQA）、SwiGLU激活函数、旋转位置编码（RoPE）以及带预归一化的RMSNorm

研究团队移除了Qwen2中使用的QKV偏置，并在注意力机制中引入QK-Norm，以确保Qwen3在训练过程中的稳定性

md都听说过，都没有深入学过，还是需要练那个手搓大模型，连着原理+代码一起练

这个QK-Norm是什么东西，https://www.zhihu.com/question/1938765666330084537

Tie Embedding又是啥

这个是权重绑定，就是说，输入的embedding和输出的embedding用的是一个权重，仅此而已

qwen3 的tokenizer依旧是BBPE，然后词表大小是151,669个token

## 预训练

看不懂，

## 后训练

Qwen3的后训练主要为了两个目标，一个是

思维控制：整合“非思考”和“思考”两种不同模式，使用户能灵活选择模型是否进行推理，并通过设定思考过程的token预算来控制推理深度

就是那个入参的thinking，

一个是强到弱知识蒸馏：优化轻量级模型的后训练流程。通过利用大规模模型的知识，显著降低小规模模型构建所需的计算资源和开发工作量。

就是用训好的那个大模型去蒸馏小模型

首先是从Base模型到最终的两个Moe的训练

然后是两个MOE蒸馏得到一系列dense小模型

首先是怎么从Base的到两个MOE的

第一步冷启动，Long-CoT

第二步推理强化学习 查询-验证器 GRPO

第三步骤 思考融合，这个我真知道，就是把think和no think融入进去用户的查询里面，然后后面的输出格式的<think>是不变的，然后做SFT

第四部就是通用的RL，针对一些通用的场景做强化学习

第二个就是这些蒸馏小模型，小模型蒸馏也分两个阶段

首先是Off-policy Distillation

就是说，off-policy就是让大模型输出，小模型直接去学，相当于预训练，只用大模型的输入和输出罐鸭子

然后on-policy就是让小模型自己输出，通过计算小模型和大模型的kl散度来去约束





