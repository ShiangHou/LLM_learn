# PPO



我们在TRPO要用到hessian矩阵，然后用共轭梯度去计算，very的复杂



PPO其实更多的是对TRPO的约束进行一个改进，用软约束去做了一下



![image-20260419155541400](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419155541400.png)

为什么不用下面那个？因为那个beta是惩罚项的惩罚系数，在不同的学习，或者学习的不同阶段里，很难去选取一个固定的合适的

## PPO-penalty

那么思路就是说，我还是想用软约束，所以就用动态的beta

![image-20260419155850162](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419155850162.png)

怎么去动态调整，？就是我们看KL散度，如下

![image-20260419160011528](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419160011528.png)

当然减小多少，增大多少，target是多少，需要经验了，就是如果超了1.5倍，我们就把beta除2

哈哈哈



这就是PPO的第一种变体

## PPO-clip

刚才的PPO-penalty是随着时间的推移去改变beta

而PPO是直接把策略的改动限制在了一个范围里面

![image-20260419160721543](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419160721543.png)

意思就是说，如果我的x落在了l和r范围内，就正常更新x，如果比l还要靠左，我就取l，如果比r还要靠右，我就取r



而这里我们需要限制的是重要性权重，也就是那个重要性采样的系数，pi的新策略除老策略，用这个约束，不用KL散度了



原论文里面那个是0.2，也就是左边界是0.8，右边界是1.2

![image-20260419161140064](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419161140064.png)

这些是深度学习包用的

![image-20260419161246345](C:\Users\13090\AppData\Roaming\Typora\typora-user-images\image-20260419161246345.png)

看不懂，妈的

## 香农熵和KL散度



衡量一个东西的惊喜度



说说什么叫做惊喜！



就是 log2（1/Px）



然后平均惊喜度就是 惊喜度的期望，也就是 -logPx * Px（负是因为1/Px





那么KL散度就是额外的香农熵



如果真实分布是X，那么Y就是衡量除了X之外的额外的香农熵



用Y的香农熵减去X的就行



然后就是额外的香农熵在X求期望就行（即再乘概率再求和）



## GAE



