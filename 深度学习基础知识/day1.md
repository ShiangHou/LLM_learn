# day1



- **理论**：重点学习反向传播、链式法则、Cross Entropy 和 Transformer 的整体流程，不只是记概念，要能解释“为什么”。
- **代码：手写一个最小的两层神经网络，完成前向传播、反向传播和参数更新；再用 PyTorch 验证梯度是否一致。
- **项目映射**：把今天学到的知识联系到你做过的 SFT、GRPO 或 Rerank 项目，思考这些原理分别对应训练链路中的哪一部分。
- **面试输出：脱离资料，回答几个问题，例如：
  - `loss.backward()` 到底做了什么？
  - 为什么需要 `optimizer.zero_grad()`？
  - Cross Entropy 为什么适合语言模型？
  - 一个 token 的 loss 是怎样传回 Transformer 参数的？

最后花 10 分钟做复盘，记录四件事：



## 一：LOSS

假设有一个模型：
$$
\hat y=f(x;\theta)
$$
其中：

- $x$：输入
- $\hat y$：模型预测
- $\theta$：模型参数
- $y$：正确答案

训练首先需要一个 loss：
$$
L(\theta)=\operatorname{Loss}(f(x;\theta),y)
$$
loss 衡量模型预测有多差。

> 我们常说的loss的形式就是这个loss函数的样子，是怎么组合的，比如极大似然就是让概率乘积最小，等



接下来要回答一个核心问题：

> 每个参数应该朝哪个方向修改，才能让 loss 下降？

对于参数 $\theta_i$，我们计算：
$$
\frac{\partial L}{\partial \theta_i}
$$
它表示：

> 参数 $\theta_i$ 发生一个很小变化时，loss 会发生多大的变化。

然后使用梯度下降：
$$
\theta_i \leftarrow \theta_i-\eta\frac{\partial L}{\partial\theta_i}
$$
其中 $\eta$ 是学习率。

为什么前面有负号？

因为梯度指向函数增长最快的方向，所以负梯度指向函数下降最快的局部方向。

整个训练过程本质上只有两步：
$$
\text{计算梯度}
\quad\longrightarrow\quad
\text{使用梯度更新参数}
$$
在 PyTorch 中分别对应：

```python
loss.backward()
optimizer.step()
```

特别注意：

> `loss.backward()` 只计算梯度，不更新参数。

### 导数、偏导数和梯度

####  导数的直觉

假设：
$$
y=x^2
$$
导数是：
$$
\frac{dy}{dx}=2x
$$
当 $x=3$ 时：
$$
\frac{dy}{dx}=6
$$
它表示在 $x=3$ 附近，$x$ 增加一个很小的量 $\Delta x$，$y$ 大约增加：
$$
\Delta y\approx 6\Delta x
$$
例如：
$$
\Delta x=0.01
$$
那么：
$$
\Delta y\approx0.06
$$
导数不是“变化量本身”，而是局部变化率。

------

####  偏导数

神经网络的 loss 通常依赖大量参数：
$$
L=L(w_1,w_2,\ldots,w_n)
$$
对其中一个参数求导时，其他参数暂时视为常数：
$$
\frac{\partial L}{\partial w_i}
$$
这叫偏导数。

例如：
$$
L=w_1^2+3w_1w_2
$$
则：
$$
\frac{\partial L}{\partial w_1}=2w_1+3w_2
$$

------

#### 梯度

将所有偏导数组合起来：
$$
\nabla L=
\begin{bmatrix}
\frac{\partial L}{\partial w_1}\\
\frac{\partial L}{\partial w_2}\\
\vdots\\
\frac{\partial L}{\partial w_n}
\end{bmatrix}
$$
这就是梯度。

梯度回答的是：

> 每一个参数分别对最终 loss 有多大影响？

神经网络训练不是只求一个导数，而是要为数十亿个参数分别计算梯度。

### 计算图

反向传播并不是一个神秘的新数学公式，它本质上是：

> 将复杂函数拆成许多简单运算，然后沿计算图反方向重复使用链式法则。

考虑：
$$
a=wx
$$
前向传播：
$$
x,w\rightarrow a\rightarrow b\rightarrow L
$$
假设：
$$
x=2,\quad w=3,\quad c=1
$$
那么：
$$
a=3\times2=6
$$
现在计算：
$$
\frac{\partial L}{\partial w}
$$
由于 $w$ 并不直接连接到 $L$，需要沿路径计算：
$$
w\rightarrow a\rightarrow b\rightarrow L
$$
所以：
$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial b}
\frac{\partial b}{\partial a}
\frac{\partial a}{\partial w}
$$
分别求局部导数：
$$
\frac{\partial L}{\partial b}=2b=14
$$
因此：
$$
\frac{\partial L}{\partial w}
=14\times1\times2=28
$$
这就是链式法则。

#### 一层复合函数

假设：
$$
y=f(u)
$$
那么：
$$
y=f(g(x))
$$
链式法则为：
$$
\frac{dy}{dx}
=
\frac{dy}{du}
\frac{du}{dx}
$$
直觉上可以理解为：
$$
x\text{ 对 }u\text{ 的影响}
\times
u\text{ 对 }y\text{ 的影响}
$$
得到：
$$
x\text{ 对 }y\text{ 的最终影响}
$$

------

####  多层复合函数

如果：
$$
x\rightarrow a\rightarrow b\rightarrow c\rightarrow L
$$
那么：
$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial c}
\frac{\partial c}{\partial b}
\frac{\partial b}{\partial a}
\frac{\partial a}{\partial x}
$$
神经网络看起来复杂，只是因为这条链非常长，并且中间有很多分支。

------

#### 分支时为什么梯度要相加

假设一个变量 $x$ 同时影响两条路径：
$$
a=2x
$$
则：
$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial a}\frac{\partial a}{\partial x}
+
\frac{\partial L}{\partial b}\frac{\partial b}{\partial x}
$$
也就是：
$$
\frac{\partial L}{\partial x}
=1\times2+1\times2x
$$
为什么要相加？

因为 $x$ 通过多条路径影响 loss，总影响等于每条路径影响之和。

这件事在 Transformer 的残差连接中非常重要。

假设：
$$
y=x+F(x)
$$
那么：
$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial y}
\left(
1+\frac{\partial F}{\partial x}
\right)
$$
残差连接提供了一条导数为 1 的直接路径：
$$
x\rightarrow y
$$
这有助于梯度向深层网络传播。

### 反向传播

反向传播可以定义为：

> 从最终 loss 出发，按照计算图的逆拓扑顺序，使用链式法则计算每个中间变量和参数的梯度。

反向传播通常分成三个概念。

#### 上游梯度

某个节点收到的梯度，例如：
$$
\frac{\partial L}{\partial y}
$$
它表示后续整个网络已经计算出的、loss 对当前输出 $y$ 的梯度。

#### 局部梯度

当前运算自己的导数，例如：
$$
y=wx
$$
局部梯度是：
$$
\frac{\partial y}{\partial w}=x
$$

####  下游梯度

根据链式法则：
$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial y}
\frac{\partial y}{\partial w}
$$
即：

> 下游梯度 = 上游梯度 × 局部梯度。

整个反向传播，就是每个计算节点重复这一步。



### 两层网络的例子

考虑一个两层分类网络：
$$
X\rightarrow \text{Linear}_1\rightarrow\operatorname{ReLU}
\rightarrow\text{Linear}_2\rightarrow\text{Softmax}
\rightarrow\text{Cross Entropy}
$$
设：

- batch size：$B$
- 输入维度：$D$
- 隐藏层维度：$H$
- 分类数：$C$

参数形状：
$$
X\in\mathbb R^{B\times D}
$$

------

##### 第一层线性变换

$$
Z_1=XW_1+b_1
$$

形状：
$$
(B,D)(D,H)\rightarrow(B,H)
$$
这里 $b_1$ 会广播到 batch 中每一个样本。

> 广播机制就是说



------

#####  ReLU

$$
A_1=\operatorname{ReLU}(Z_1)
$$

其中：
$$
\operatorname{ReLU}(z)=\max(0,z)
$$
导数：
$$
\frac{\partial\operatorname{ReLU}(z)}{\partial z}
=
\begin{cases}
1,&z>0\\
0,&z<0
\end{cases}
$$
当 $z=0$ 时不可导，工程中通常定义为 0。

ReLU 的作用不仅是“把负数变成 0”。

更重要的是：

> 它引入非线性。

如果没有激活函数：
$$
XW_1W_2
$$
仍然只是一个线性变换：
$$
XW'
$$
无论叠多少个纯线性层，都等价于一个线性层。

------

##### 第二层线性变换

$$
Z_2=A_1W_2+b_2
$$

$Z_2$ 也叫 logits。

注意：

> logits 不是概率。

logits 可以是任意实数，例如：
$$
[2.1,-0.7,4.3]
$$

------

##### Softmax

Softmax 将 logits 转成概率：
$$
p_j=\frac{e^{z_j}}{\sum_{k=1}^{C}e^{z_k}}
$$
它保证：
$$
0<p_j<1
$$
并且：
$$
\sum_jp_j=1
$$
因此可以把输出解释成分类概率分布。

实际实现需要先减去最大值：
$$
p_j=
\frac{e^{z_j-\max(z)}}
{\sum_ke^{z_k-\max(z)}}
$$
这是为了防止指数运算溢出。

因为 Softmax 对所有 logits 同时加减同一个常数不会改变结果：
$$
\operatorname{softmax}(z)
=
\operatorname{softmax}(z-c)
$$

> 这里的意思是，比如分类是3类，hidden_dim是1024，那么我们会在logits之前弄一个1024*3的线性层，搞成3维，最后的logits就是3维度的，然后再softmax

#### Cross Entropy

假设正确类别为 $y$，模型对正确类别给出的概率为 $p_y$。

Cross Entropy loss：
$$
L=-\log p_y
$$
例如正确答案是类别 2：

| 模型给正确类别的概率 | loss  |
| -------------------- | ----- |
| 0.9                  | 0.105 |
| 0.5                  | 0.693 |
| 0.1                  | 2.303 |
| 0.01                 | 4.605 |

这意味着：

- 正确类别概率越高，loss 越小。
- 正确类别概率越低，loss 越大。
- 模型非常自信但预测错误时，会受到很大惩罚。

------

#### 为什么使用负对数

直接最大化正确类别概率：
$$
\max p_y
$$
等价于最大化其对数：
$$
\max \log p_y
$$
又等价于最小化负对数：
$$
\min -\log p_y
$$
使用对数有几个重要原因。

第一，序列概率是多个 token 条件概率的乘积：
$$
P(y_1,\ldots,y_T\mid x)
=
\prod_{t=1}^{T}P(y_t\mid x,y_{<t})
$$
取对数后：
$$
\log P(y_1,\ldots,y_T\mid x)
=
\sum_{t=1}^{T}\log P(y_t\mid x,y_{<t})
$$
乘积变成加法，更容易计算，也更数值稳定。

第二，负对数会强烈惩罚“高置信度错误”。

第三，最大似然估计天然会得到负对数似然，而分类问题中的负对数似然正是 Cross Entropy。

#### Softmax + Cross Entropy 的关键梯度

这是今天最值得手推的一条公式。

设 logits 为：
$$
z=[z_1,z_2,\ldots,z_C]
$$
Softmax：
$$
p_j=\frac{e^{z_j}}{\sum_ke^{z_k}}
$$
正确类别为 $y$，loss：
$$
L=-\log p_y
$$
展开：
$$
L
=
-\log
\frac{e^{z_y}}{\sum_ke^{z_k}}
$$
对任意 logit $z_j$ 求导：
$$
\frac{\partial L}{\partial z_j}
=
-\mathbf 1[j=y]
+
\frac{e^{z_j}}{\sum_ke^{z_k}}
$$
> 这里是吧上面的log拆开，变成-loge z +logsume，求导后就是-1（因为logez就是z，）后面就是log的导



因此：
$$
\boxed{
\frac{\partial L}{\partial z_j}
=
p_j-\mathbf 1[j=y]
}
$$
写成向量：
$$
\boxed{
\frac{\partial L}{\partial z}
=
p-\operatorname{onehot}(y)
}
$$
这是一个非常漂亮的结果。

假设：
$$
p=[0.7,0.2,0.1]
$$
正确类别是第二类：
$$
\operatorname{onehot}(y)=[0,1,0]
$$
那么：
$$
\frac{\partial L}{\partial z}
=
[0.7,-0.8,0.1]
$$
梯度下降时：
$$
z\leftarrow z-\eta\frac{\partial L}{\partial z}
$$
因此：

- 第一类 logit 会下降，因为梯度是正的。
- 第二类正确 logit 会上升，因为梯度是负的。
- 第三类 logit 会下降。

这正是我们想要的效果。

------

### 两层网络的反向传播推导

前向过程：
$$
Z_1=XW_1+b_1
$$
为了简化记号，记：
$$
dZ_2=\frac{\partial L}{\partial Z_2}
$$
对于 batch 平均 loss：
$$
dZ_2=
\frac{P-\operatorname{onehot}(y)}{B}
$$

------

### 第二层参数梯度

因为：
$$
Z_2=A_1W_2+b_2
$$
所以：
$$
\boxed{
dW_2=A_1^\top dZ_2
}
$$
对隐藏层输出的梯度：
$$
\boxed{
dA_1=dZ_2W_2^\top
}
$$
理解矩阵转置不要只靠记忆，可以通过形状检查。

已知：
$$
A_1^\top\in\mathbb R^{H\times B}
$$
因此：
$$
dW_2\in\mathbb R^{H\times C}
$$
刚好与 $W_2$ 形状一致。

------

###  ReLU 梯度

$$
A_1=\operatorname{ReLU}(Z_1)
$$

所以：
$$
\boxed{
dZ_1=dA_1\odot\mathbf 1[Z_1>0]
}
$$
其中 $\odot$ 表示逐元素相乘。

如果某个位置：
$$
Z_{1,ij}<0
$$
那么 ReLU 输出恒为 0，该位置局部梯度为 0，梯度不会继续向前传播。

------

### 第一层参数梯度

因为：
$$
Z_1=XW_1+b_1
$$
所以：
$$
\boxed{
dW_1=X^\top dZ_1
}
$$
对输入的梯度是：
$$
dX=dZ_1W_1^\top
$$
一般训练数据 $X$ 不需要更新，因此通常不会关心 $dX$。但在更复杂的网络中，$X$ 可能是上一层的输出，所以必须继续计算。

## 代码：手写最小两层神经网络

下面的代码完整实现：

- 前向传播
- Softmax
- Cross Entropy
- 反向传播
- 参数更新
- PyTorch 梯度验证

```python
import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 1. 构造数据
# ============================================================

rng = np.random.default_rng(42)#随机数生成器，应该是np的类，42是随机种子

batch_size = 4
input_dim = 3
hidden_dim = 5
num_classes = 3

# 输入：4 个样本，每个样本 3 个特征
X = rng.normal(size=(batch_size, input_dim))#应该也是np的那个类的方法，入参数是一个tup，会返回，然后这里的normal就是从正态分布中采样

# 每个样本的分类标签
y = np.array([0, 2, 1, 2], dtype=np.int64)# 一个array，数据类型


# ============================================================
# 2. 初始化参数
# ============================================================

W1 = rng.normal(scale=0.1, size=(input_dim, hidden_dim))#也是随机生成，这个scale是
b1 = np.zeros(hidden_dim)

W2 = rng.normal(scale=0.1, size=(hidden_dim, num_classes))#scale表示正态分布的标准差是0.1，
b2 = np.zeros(num_classes)#全0的，这个的是(5,)的维度


# ============================================================
# 3. 前向传播
# ============================================================

# 第一层线性变换
z1 = X @ W1 + b1#
#X的维度是(4,3)，W1的是（3,5),乘完后是(4,5)然后b1的是(5,)，直接加的话，根据广播机制会加到每一行

# ReLU
a1 = np.maximum(z1, 0.0)#普通的取max的，就是把整个(4,5)矩阵中所有小于0的全部搞成0

# 第二层线性变换，得到 logits
logits = a1 @ W2 + b2#W2是(5,3)的维度

# 数值稳定的 Softmax
shifted_logits = logits - np.max(logits, axis=1, keepdims=True)#这里是找到每一行的最大值，因为axis是1，所以是按列比较，也就是每一行找一个最大，这样其实是(3,),keepdim后就是(3,1)
exp_logits = np.exp(shifted_logits)#简单的求指数
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Cross Entropy
correct_probs = probs[np.arange(batch_size), y]
#这一步的目的是要从probs里面拿到每个样本的答案，前面的np.arange(batch_size)就是取了0 1 2 3，后面的y就是标准答案，因为probs本身是一个(4, 3)的，这里相当于probs[0],probs[1]probs[2]probs[3]是4个样本，然后当我们取，比如probs[0]的时候，拿到了一个3维度的模型输出的，然后这个时候，假设y是每个样本的真是标签，这里是[0,2,1,2]相当于，我取第一个样本是probs[0]，然后第0个样本的正确是0，那么我就拿到模型输出的第一个样本的第0个预测的概率（第0个是正确答案，因为[0,2,1,2]是每个样本的正确，第0个正好是0，最后，我们得到了一个这4个样本的正确答案的loss
#相当于correct_probs取[]
loss = -np.mean(np.log(correct_probs))#计算交叉熵，后面的就是pi，即第i个样本预测的真实标签的概率，然后log取和后平均到batch上

print("Manual loss:", loss)


# ============================================================
# 4. 反向传播
# ============================================================

# Softmax + Cross Entropy 的梯度：
# dlogits = (probs - one_hot(y)) / batch_size

dlogits = probs.copy()
dlogits[np.arange(batch_size), y] -= 1.0#这个就是之前的，拿到在真实样本上的概率，然后减去1，就是probs - one_hot(y)
dlogits /= batch_size#除batch

# 第二层梯度
dW2 = a1.T @ dlogits

#上一层的梯度是dlogits，下一层是AW，要对W求梯度，就是本身的梯度（at）和上一层的dlogits相乘

#这个事对bias的梯度。直接求和就行，因为上一步的事dlogits*b，求导后就是dlogits
db2 = np.sum(dlogits, axis=0)

# 梯度传回隐藏层

#这个是隐藏层激活值的梯度
da1 = dlogits @ W2.T


#总结一下，上面的分别是一个 aW+b，然后我们对a，对W，对b求梯度，记得乘上游的梯度
=====


# ReLU 反向传播
#这个是来算a1的，在forward的时候，a1=ReLU(z1)

dz1 = da1 * (z1 > 0)#z1是我们之前的，相当于吧

# 第一层梯度
#上一层返回的是dz1

dW1 = X.T @ dz1
db1 = np.sum(dz1, axis=0)


# ============================================================
# 5. 使用 PyTorch 验证梯度
# ============================================================

# 使用 float64，减少数值误差
X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.long)

W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)#设置true，反向的时候会存在W1_t.grad里面


b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)

W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

z1_t = X_t @ W1_t + b1_t
a1_t = torch.relu(z1_t)
logits_t = a1_t @ W2_t + b2_t
#前向传播


loss_t = F.cross_entropy(logits_t, y_t)

# 自动反向传播
loss_t.backward()

print("PyTorch loss:", loss_t.item())

print(
    "W1 gradient max error:",
    np.max(np.abs(dW1 - W1_t.grad.detach().numpy()))
)

print(
    "b1 gradient max error:",
    np.max(np.abs(db1 - b1_t.grad.detach().numpy()))
)

print(
    "W2 gradient max error:",
    np.max(np.abs(dW2 - W2_t.grad.detach().numpy()))
)

print(
    "b2 gradient max error:",
    np.max(np.abs(db2 - b2_t.grad.detach().numpy()))
)


# ============================================================
# 6. 参数更新
# ============================================================

learning_rate = 0.1

W1 -= learning_rate * dW1
b1 -= learning_rate * db1

W2 -= learning_rate * dW2
b2 -= learning_rate * db2


# ============================================================
# 7. 更新后重新计算 loss
# ============================================================

z1_new = X @ W1 + b1
a1_new = np.maximum(z1_new, 0.0)
logits_new = a1_new @ W2 + b2

shifted_new = logits_new - np.max(
    logits_new,
    axis=1,
    keepdims=True
)

exp_new = np.exp(shifted_new)
probs_new = exp_new / np.sum(exp_new, axis=1, keepdims=True)

loss_new = -np.mean(
    np.log(probs_new[np.arange(batch_size), y])
)

print("Loss after one update:", loss_new)
```

正常情况下，四个梯度误差应该非常小，通常在：

```
1e-15 ～ 1e-8
```

这说明：

> 你手写的反向传播，与 PyTorch autograd 计算出的梯度一致。

------

### 逐行理解这段反向传播

最核心的一行：

```
dlogits = probs.copy()
dlogits[np.arange(batch_size), y] -= 1.0
dlogits /= batch_size
```

它对应：
$$
\frac{\partial L}{\partial z}
=
\frac{p-\operatorname{onehot}(y)}{B}
$$
为什么除以 batch size？

因为 loss 是 batch 内样本 loss 的平均：
$$
L=\frac{1}{B}\sum_{i=1}^{B}L_i
$$
所以每个样本产生的梯度也要除以 $B$。

------

这一行：

```
dW2 = a1.T @ dlogits
```

对应：
$$
dW_2=A_1^\top dZ_2
$$
可以理解为：

> 输入到第二层的每一个隐藏特征，与输出误差信号相乘，再对 batch 求和。

------

这一行：

```
da1 = dlogits @ W2.T
```

对应：
$$
dA_1=dZ_2W_2^\top
$$
它将分类输出处的误差信号，沿第二层权重传回隐藏层。

------

这一行：

```
dz1 = da1 * (z1 > 0)
```

对应 ReLU 的局部导数。

对于前向时被 ReLU 截断成 0 的位置，梯度也被截断。

------

参数更新：

```
W1 -= learning_rate * dW1
```

对应：
$$
W_1\leftarrow W_1-\eta dW_1
$$
这才是真正改变模型参数的步骤。

------

### PyTorch 的自动微分做了什么

考虑下面代码：

```
x = torch.tensor(2.0, requires_grad=True)
w = torch.tensor(3.0, requires_grad=True)

a = x * w
b = a + 1
loss = b ** 2

loss.backward()

print(w.grad)
```

前向时，PyTorch 不只是计算结果，还构建了一个动态计算图：

```
x ----\
       multiply -> a -> add -> b -> square -> loss
w ----/              ^
                     |
                     1
```

每个计算节点会保存：

- 当前运算是什么。
- 哪些 tensor 是输入。
- 反向传播时需要哪些中间结果。
- 如何计算局部梯度。

执行：

```
loss.backward()
```

PyTorch 会：

1. 将 loss 对自身的梯度初始化为 1：
   $$
   \frac{\partial L}{\partial L}=1
   $$

2. 按照计算图的逆拓扑顺序遍历节点。

3. 对每个节点调用对应的 backward 规则。

4. 使用链式法则将梯度传给父节点。

5. 将叶子参数的梯度累积到 `.grad` 中。

所以：

```
w.grad
```

保存：
$$
\frac{\partial L}{\partial w}
$$

------

### `loss.backward()` 到底做了什么

面试中可以这样回答：

> `loss.backward()` 会从 loss 节点出发，沿前向过程中构建的动态计算图反向遍历。对每个算子，它利用该算子的局部导数和上游梯度，通过链式法则计算输入梯度。对于 `requires_grad=True` 的叶子张量，例如模型参数，最终梯度会累积到参数的 `.grad` 属性中。它只负责计算和累积梯度，并不会更新参数，参数更新由 `optimizer.step()` 完成。

可以继续补充四个细节。

#### 14.1 loss 通常必须是标量

如果 loss 是标量，PyTorch 默认：
$$
\frac{\partial L}{\partial L}=1
$$
如果输出不是标量，需要显式提供上游梯度：

```
y.backward(gradient=torch.ones_like(y))
```

------

#### 14.2 梯度存入 `.grad`

```
for name, param in model.named_parameters():
    print(name, param.grad)
```

在 `backward()` 之前通常是：

```
None
```

执行之后才会得到具体梯度。

------

#### 14.3 梯度默认累积

如果连续执行：

```
loss.backward()
loss.backward()
```

那么第二次并不会覆盖第一次，而是：
$$
\text{grad}_{new}
=
\text{grad}_{old}
+
\text{grad}_{current}
$$

------

#### 14.4 计算图通常在 backward 后释放

默认情况下：

```
loss.backward()
```

之后中间计算图会被释放以节省显存。

如果确实需要对同一张图多次 backward，可以：

```
loss.backward(retain_graph=True)
```

但滥用会显著增加显存占用。

------

#### 为什么需要 `optimizer.zero_grad()`

标准训练循环：

```python
for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(batch["input_ids"])
    loss = compute_loss(outputs, batch["labels"])

    loss.backward()
    optimizer.step()
```

`zero_grad()` 的作用是清空上一次迭代保存在参数 `.grad` 中的梯度。

因为 PyTorch 默认执行：
$$
\text{param.grad}
\mathrel{+}=
\text{current gradient}
$$
而不是：
$$
\text{param.grad}
=
\text{current gradient}
$$
如果不清空，假设：

第一步梯度：
$$
g_1
$$
第二步梯度：
$$
g_2
$$
第二步实际使用的会变成：
$$
g_1+g_2
$$
而不是单独的 $g_2$。

这可能导致：

- 梯度越来越大。
- 更新方向混入历史 batch。
- loss 波动甚至训练发散。
- 实际 batch size 与预期不一致。

------

#### 为什么 PyTorch 要设计成累积，而不是自动覆盖

因为梯度累积本身是一项重要功能。

例如显存只能放一个 micro-batch，但希望有效 batch size 为 8：

```python
optimizer.zero_grad()

for micro_step in range(8):
    loss = model(batch[micro_step]) / 8
    loss.backward()

optimizer.step()
```

这时：
$$
g=
\frac{1}{8}
\sum_{i=1}^{8}g_i
$$
等价于对更大的 batch 求平均梯度。

因此 PyTorch 的设计是：

> 默认累积，由使用者决定何时清空。

------

####  `zero_grad(set_to_none=True)`

常见写法：

```
optimizer.zero_grad(set_to_none=True)
```

它不是把梯度 tensor 全部填成 0，而是把 `.grad` 设置为 `None`。

通常可以减少内存写入，性能略好，也是现代 PyTorch 常用设置。

## 十六、Transformer 的完整前向流程

对于因果语言模型，例如 Qwen、Llama，整体流程是：

```
文本
↓
Tokenizer
↓
input_ids
↓
Token Embedding
↓
多层 Transformer Block
↓
Final LayerNorm
↓
LM Head
↓
每个位置上的词表 logits
↓
Cross Entropy
```

------

## 16.1 Tokenizer

假设文本：

```
我喜欢机器学习
```

Tokenizer 可能转换为：

```
[1052, 8432, 19873, 617]
```

这些整数是 token ID。

模型本身不直接处理字符串，而是处理 token ID。

------

## 16.2 Embedding

Embedding 矩阵：
$$
E\in\mathbb R^{V\times d}
$$
其中：

- $V$：词表大小
- $d$：隐藏维度

对于 token ID $i$，取出：
$$
E_i
$$
作为该 token 的初始向量表示。

输入序列长度为 $T$，则：
$$
X\in\mathbb R^{T\times d}
$$
实际还会包含 batch 维度：
$$
X\in\mathbb R^{B\times T\times d}
$$

------

### 十七、Transformer Block

现代 Decoder-only Transformer 通常采用 Pre-Norm 结构：
$$
X'=X+\operatorname{Attention}(\operatorname{LN}(X))
$$
一个 block 主要包含：

1. LayerNorm 或 RMSNorm
2. Self-Attention
3. 残差连接
4. MLP
5. 第二个残差连接

------

# 十八、Self-Attention 的完整过程

输入：
$$
X\in\mathbb R^{T\times d}
$$
通过三个线性层：
$$
Q=XW_Q
$$
其中：

- Query：当前位置想查询什么。
- Key：每个位置提供什么索引信息。
- Value：每个位置真正携带的内容。

注意力分数：
$$
S=\frac{QK^\top}{\sqrt{d_k}}
$$
再加 causal mask：
$$
S'=S+M
$$
其中未来位置对应：
$$
M_{ij}=-\infty,\quad j>i
$$
然后：
$$
A=\operatorname{softmax}(S')
$$
输出：
$$
O=AV
$$
再经过输出投影：
$$
Y=OW_O
$$

------

## 18.1 为什么除以 $\sqrt{d_k}$

如果 $Q$ 和 $K$ 中各维大致独立、方差为 1，那么内积：
$$
Q\cdot K
$$
的方差会随维度 $d_k$ 增大。

分数过大后，Softmax 会变得非常尖锐：

```
[100, 10, -30]
```

经过 Softmax 后几乎变成：

```
[1, 0, 0]
```

这时梯度容易过小。

除以：
$$
\sqrt{d_k}
$$
可以让注意力分数保持在更合理的数值范围。

------

## 18.2 为什么需要 causal mask

语言模型在位置 $t$ 预测下一个 token 时，只能看到：
$$
x_1,\ldots,x_t
$$
不能看到未来 token：
$$
x_{t+1},x_{t+2},\ldots
$$
否则训练时模型会直接偷看答案。

例如：

```
输入：北京是中国的
标签：京是中国的首
```

在预测“首”时，只能使用“北京是中国的”，不能提前看到“首”。

------

## 18.3 为什么需要残差连接

残差结构：
$$
Y=X+F(X)
$$
它有两个作用。

第一，允许网络保留原始信息。

如果某一层学习不到有价值的变换，可以令：
$$
F(X)\approx0
$$
那么：
$$
Y\approx X
$$
第二，提供更直接的梯度传播路径。
$$
\frac{\partial Y}{\partial X}
=
I+\frac{\partial F}{\partial X}
$$
其中 $I$ 是恒等路径，减少深层网络中梯度完全消失的风险。

------

## 18.4 为什么需要 LayerNorm 或 RMSNorm

随着网络层数增加，中间表示的数值范围可能不断变化。

Normalization 的作用包括：

- 稳定激活值范围。
- 改善优化条件。
- 让训练对初始化和学习率更稳定。
- 降低深层模型训练发散概率。

RMSNorm 与 LayerNorm 的主要区别是，RMSNorm 通常不减去均值，只根据均方根进行缩放。

# 十九、MLP 在 Transformer 中做什么

典型 MLP：
$$
H=\phi(XW_{\text{up}})
$$
很多现代模型使用门控结构，例如 SwiGLU：
$$
H=
\operatorname{SiLU}(XW_{\text{gate}})
\odot
(XW_{\text{up}})
$$
Attention 更偏向：

> 在 token 之间交换和聚合信息。

MLP 更偏向：

> 对每个 token 的特征进行非线性变换和知识提取。

两者都是 Transformer 能力的重要来源。

------

# 二十、从隐藏状态到词表概率

经过最后一层 Transformer 后，得到：
$$
H\in\mathbb R^{B\times T\times d}
$$
通过 LM Head：
$$
Z=HW_{\text{vocab}}+b
$$
其中：
$$
W_{\text{vocab}}\in\mathbb R^{d\times V}
$$
得到：
$$
Z\in\mathbb R^{B\times T\times V}
$$
这意味着：

> 对 batch 中每个样本、每个 token 位置，模型都会输出一个长度为词表大小 $V$ 的 logits 向量。

例如：

```
位置 0：对整个词表的 logits
位置 1：对整个词表的 logits
位置 2：对整个词表的 logits
...
```

------

# 二十一、语言模型的 label shift

假设 token 序列：

```
[BOS, 我, 喜欢, 机器, 学习, EOS]
```

训练时：

```
输入：
[BOS, 我, 喜欢, 机器, 学习]

标签：
[我, 喜欢, 机器, 学习, EOS]
```

也就是：
$$
x_t\rightarrow y_t=x_{t+1}
$$
模型在每个位置预测下一个 token。

总 loss：
$$
L=
\frac{1}{T}
\sum_{t=1}^{T}
-\log P(y_t\mid x_{\le t})
$$
因此语言模型训练本质上可以看成：

> 在每个 token 位置上做一次词表大小的多分类。

------

# 二十二、为什么 Cross Entropy 适合语言模型

面试可以这样回答：

> 因果语言模型的目标是在给定历史 token 的条件下，预测下一个 token。下一个 token 是词表中的一个离散类别，因此模型需要输出一个词表上的概率分布。Softmax 可以将 logits 转换为分类概率，而 Cross Entropy 等价于正确 token 的负对数似然。最小化 token-level Cross Entropy，就等价于最大化训练语料在模型下的条件似然，所以它天然适合语言模型的最大似然训练。

更深入地说，有四个原因。

## 22.1 下一个 token 是多分类问题

词表大小假设为：
$$
V=150000
$$
每个位置需要从 15 万个 token 中选择一个。

因此是标准多分类。

------

## 22.2 Cross Entropy 对应最大似然估计

模型希望最大化：
$$
P_\theta(y_t\mid x_{\le t})
$$
对整个序列：
$$
\max_\theta
\prod_t
P_\theta(y_t\mid x_{\le t})
$$
取负对数后变成：
$$
\min_\theta
-\sum_t
\log P_\theta(y_t\mid x_{\le t})
$$
这正是 token-level Cross Entropy。

------

## 22.3 它是可微的

虽然正确 token 是离散的，但模型输出的是连续 logits。

Cross Entropy 对 logits 的梯度非常简单：
$$
\frac{\partial L}{\partial z}
=
p-\operatorname{onehot}(y)
$$
可以稳定地通过 Transformer 反向传播。

------

## 22.4 它对概率分布提供完整监督

不是只告诉模型“答案对还是错”，而是对整个词表分布产生梯度：

- 正确 token 的 logit 提高。
- 所有错误 token 的 logit 根据当前概率降低。

------

# 二十三、Cross Entropy 的局限性

Cross Entropy 很适合语言建模，但它并不等于最终业务目标。

它存在几个重要问题。

## 23.1 Teacher Forcing 与推理不一致

训练时模型总是看到正确历史：
$$
P(y_t\mid y_{<t})
$$
推理时模型看到的是自己生成的历史：
$$
P(\hat y_t\mid \hat y_{<t})
$$
早期错误可能不断累积，这叫 exposure bias。

------

## 23.2 token loss 不完全等于序列质量

两个回答可能 token-level likelihood 接近，但整体质量差异很大。

例如：

- 是否遵循 JSON 格式。
- 是否真正解决用户问题。
- 是否有事实错误。
- 是否满足长度限制。
- 推荐理由是否可信。
- Rerank 顺序是否合理。

这些都很难完全通过普通 CE 表达。

这也是为什么会使用：

- DPO
- PPO
- GRPO
- Reward Model
- Sequence-level evaluation

------

# 二十四、一个 token 的 loss 如何传回 Transformer 参数

这是面试重点。

假设模型在位置 $t$ 应该预测 token $y_t$。

模型输出 logits：
$$
z_t\in\mathbb R^V
$$
Cross Entropy：
$$
L_t=-\log P(y_t\mid x_{\le t})
$$
第一步，对 logits 求梯度：
$$
\frac{\partial L_t}{\partial z_t}
=
p_t-\operatorname{onehot}(y_t)
$$
这个梯度会推动：

- 正确 token 的 logit 上升。
- 其他 token 的 logit 下降。

------

## 24.1 从 logits 传到 LM Head

$$
z_t=h_tW_{\text{vocab}}+b
$$

因此：
$$
\frac{\partial L_t}{\partial W_{\text{vocab}}}
=
h_t^\top
\frac{\partial L_t}{\partial z_t}
$$
同时：
$$
\frac{\partial L_t}{\partial h_t}
=
\frac{\partial L_t}{\partial z_t}
W_{\text{vocab}}^\top
$$
这样梯度从词表 logits 传回最终隐藏状态 $h_t$。

------

## 24.2 传过最后的 Norm

最终隐藏状态通常经过 RMSNorm 或 LayerNorm。

梯度会根据该归一化算子的导数：

- 传回输入隐藏状态。
- 计算归一化缩放参数的梯度。

------

## 24.3 传过残差连接

假设：
$$
h=x+F(x)
$$
那么从 $h$ 收到的梯度，会分成两条路径：
$$
\frac{\partial L}{\partial x}
=
\frac{\partial L}{\partial h}
+
\frac{\partial L}{\partial h}
\frac{\partial F}{\partial x}
$$
一条直接经过残差分支，另一条进入 Attention 或 MLP。

------

## 24.4 传过 MLP

如果：
$$
F(x)=\operatorname{SiLU}(xW_1)W_2
$$
则梯度依次经过：

```
输出
↓
W2
↓
SiLU
↓
W1
↓
输入
```

同时得到：
$$
\frac{\partial L}{\partial W_2}
$$
和：
$$
\frac{\partial L}{\partial W_1}
$$

------

## 24.5 传过 Attention

Attention 输出：
$$
O=\operatorname{softmax}
\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right)V
$$
梯度会传向：

- $V$：哪些内容被聚合。
- 注意力权重：应该更多关注哪些 token。
- $Q$：当前位置应该如何查询。
- $K$：历史 token 应该如何被匹配。
- $W_Q,W_K,W_V,W_O$：注意力投影参数。

由于位置 $t$ 的隐藏状态会关注前面的 token，所以：

> 位置 $t$ 的 token loss 不仅会更新位置 $t$ 的表示，也会通过 Attention 影响位置 $0\sim t$ 的历史 token 表示。

但由于 causal mask，它不会通过当前 Attention 影响未来位置。

------

## 24.6 一直传回 Embedding

最终梯度会传到输入 token 的 embedding 向量。

由于 embedding 查表只使用了部分 token ID，所以只有本 batch 中出现过的 token 对应 embedding 行会直接收到梯度。

完整路径可以概括为：

```
token Cross Entropy
↓
logits
↓
LM Head
↓
最终隐藏状态
↓
最后一层 Transformer
↓
倒数第二层 Transformer
↓
...
↓
第一层 Transformer
↓
Token Embedding
```

同时，每一层的：

- Attention 权重
- MLP 权重
- Norm 参数
- 投影矩阵

都会获得梯度。

------

# 二十五、为什么一个 token 能更新大量参数

一个 token 的 loss 是一个标量，但它依赖整个计算过程。

例如：
$$
L_t
=
f(
W_{\text{embed}},
W_Q^{(1)},
W_K^{(1)},
W_V^{(1)},
\ldots,
W_{\text{MLP}}^{(N)},
W_{\text{vocab}}
)
$$
只要某个参数影响了 $L_t$，就可以计算：
$$
\frac{\partial L_t}{\partial W}
$$
因此一个 token 的 loss 可以为整个 Transformer 中大量参数提供梯度。

不过有些梯度可能：

- 非常小。
- 因 mask 不存在。
- 因激活函数为 0。
- 因参数被冻结而不保存。
- 因某条计算路径未参与当前输出而为 0。

------

# 二十六、映射到你的 SFT 项目

你做 Qwen SFT 时，最核心的 loss 通常仍然是：
$$
L_{\text{SFT}}
=
-\frac{1}{N}
\sum_{t\in\mathcal M}
\log
P_\theta(y_t\mid x,y_{<t})
$$
其中 $\mathcal M$ 是参与 loss 计算的 token 位置。

------

## 26.1 Prompt token 是否计算 loss

指令数据通常包括：

```
System
User
Assistant
```

常见训练方式是：

```
System/User token：label = -100
Assistant token：label = 真实 token ID
```

PyTorch Cross Entropy 中：

```
ignore_index = -100
```

因此 prompt token 不参与 loss。

模型仍然会对 prompt 做前向传播，因为 Assistant token 的预测依赖 prompt 的隐藏状态。

但是：

> prompt 位置自身的 next-token loss 可以被 mask 掉。

例如：

```
输入：
用户：推荐一只基金
助手：可以关注……

labels：
[-100, -100, -100, 可以, 关注, ……]
```

------

## 26.2 SFT 中一个样本的 loss

假设 Assistant 部分有 $T$ 个有效 token：
$$
L=
-\frac{1}{T}
\sum_{t=1}^{T}
\log P_\theta(y_t\mid x,y_{<t})
$$
长回答会产生更多 token loss。

框架通常会对所有非 `-100` token 求平均，而不是先对每个样本求平均。因此数据长度分布可能影响训练权重。

这也是你做长文本数据、packing 和 cutoff 时需要关注的问题。

------

## 26.3 LoRA 中梯度传到哪里

如果使用 LoRA：
$$
W'=W+\Delta W
$$
其中：
$$
\Delta W=BA
$$
基础参数 $W$ 被冻结：

```
W.requires_grad = False
```

只有 LoRA 参数 $A,B$ 可训练。

反向传播仍然会经过完整 Transformer，但最终只为可训练参数保存和更新梯度：
$$
\frac{\partial L}{\partial A}
$$
而基础权重不会被 optimizer 更新。

因此 LoRA 并不是“不做完整反向传播”，而是：

> 梯度仍需要穿过基础网络，只是不计算或保存被冻结参数的更新梯度。

------

## 26.4 梯度累积对应什么

你在 LLaMA-Factory 中设置：

```
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
```

表示每张卡每次只能处理 1 个样本，但累计 8 个 micro-batch 的梯度后再更新。

伪代码：

```
optimizer.zero_grad()

for i in range(8):
    loss = model(batch_i) / 8
    loss.backward()

optimizer.step()
```

如果有 8 张卡，粗略全局 batch size 为：
$$
1\times8\times8=64
$$
还需要考虑：

- 数据并行卡数。
- gradient accumulation。
- sequence packing。
- 每个 batch 的有效 token 数。

------

## 26.5 packing 为什么可能影响效果

从梯度角度看，packing 会将多个短样本拼入同一个序列。

理论上，如果：

- attention mask 正确。
- position ids 正确。
- loss mask 正确。
- 样本之间不会互相看到。
- token 加权方式一致。

那么 packing 不应大幅改变优化目标。

但实践中可能出现：

1. 样本边界 attention mask 不正确。
2. 不同样本之间发生信息泄漏。
3. EOS 处理不正确。
4. position id 逻辑与模型不匹配。
5. 长短样本的 token 权重发生变化。
6. 每步有效 token 数变化，导致实际梯度尺度变化。
7. 对生成式推荐这种格式敏感任务，拼接边界影响训练稳定性。

因此你之前观察到 0.8B 模型 `packing=true` 效果变差，可以从以下角度排查：

```
样本隔离
→ attention mask
→ EOS
→ labels mask
→ position ids
→ 每步有效 token 数
→ 实际学习率和梯度范数
```

------

# 二十七、映射到 GRPO

GRPO 不再直接把“标准答案 token”作为唯一学习信号，而是：

1. 对同一个 prompt 采样多个回答。
2. 使用 reward function 给回答打分。
3. 在组内计算相对 advantage。
4. 用 advantage 调整回答中 token 的 log probability。

一个简化目标可以写为：
$$
L_{\text{policy}}
=
-\frac{1}{G}
\sum_{i=1}^{G}
A_i
\sum_{t=1}^{T_i}
\log\pi_\theta(y_{i,t}\mid x,y_{i,<t})
$$
其中：

- $G$：同一个 prompt 的采样回答数。
- $A_i$：第 $i$ 个回答的 advantage。
- $\pi_\theta$：当前策略模型。

------

## 27.1 advantage 起什么作用

如果：
$$
A_i>0
$$
那么最小化：
$$
-A_i\log\pi_\theta(y_i)
$$
会提高该回答的概率。

如果：
$$
A_i<0
$$
则会降低该回答的概率。

所以 advantage 相当于：

> 告诉模型，这条采样轨迹中的 token 概率应该整体提高还是降低。

------

## 27.2 reward 是否直接反向传播

通常不直接。

例如 reward 来自：

- 格式规则。
- 字符串匹配。
- 外部评测脚本。
- Reward Model。
- 业务指标。
- 判别器服务。

这些 reward 函数可能不可微。

GRPO 使用策略梯度思想：
$$
\nabla_\theta J
\approx
A\nabla_\theta\log\pi_\theta(y\mid x)
$$
也就是说：

> reward 不需要对模型参数可微，只需要变成 advantage，再作为 log probability 梯度的权重。

------

## 27.3 GRPO 中梯度如何传回模型

对某个生成 token：
$$
\log\pi_\theta(y_t\mid x,y_{<t})
$$
它来自 Softmax 后正确 token 的 log probability。

因此梯度路径仍然是：

```
GRPO policy loss
↓
token log probability
↓
logits
↓
LM Head
↓
Transformer
↓
可训练参数
```

与 SFT 的区别主要在最上层监督信号：

SFT：
$$
-\log P(y_t^{\text{GT}})
$$
GRPO：
$$
-A\log P(y_t^{\text{sample}})
$$
SFT 告诉模型：

> 标准答案中的这个 token 应该提高概率。

GRPO 告诉模型：

> 这次采样回答整体表现好或差，因此其 token 概率应该相应提高或降低。

------

## 27.4 KL 项对应什么

为了防止策略模型偏离参考模型过远，常加入：
$$
L_{\text{KL}}
=
\beta
D_{\text{KL}}
(
\pi_\theta
\|
\pi_{\text{ref}}
)
$$
总目标大致为：
$$
L=L_{\text{policy}}+\beta L_{\text{KL}}
$$
KL 太大时会限制模型学习。

KL 太小时模型可能：

- 奖励黑客。
- 格式崩坏。
- 语言能力退化。
- 输出分布迅速偏离基础模型。

------

# 二十八、映射到你的 Rerank 项目

你现在的目标是：

> 对 query 和召回 item-chunk 做相关性判断，并生成理由，不再负责排序。

这里至少有三种训练方式。

------

## 28.1 纯生成式 SFT

输入：

```
query + 候选 item-chunk + 指令
```

输出：

```
{
  "relevant": true,
  "reason": "..."
}
```

训练 loss：
$$
L_{\text{gen}}
=
-\sum_t\log P(y_t\mid x,y_{<t})
$$
优点：

- 架构简单。
- 一个模型同时输出判断和理由。
- 可以直接使用 LLaMA-Factory、MS-Swift 等框架。

缺点：

- 相关性判断没有独立优化目标。
- 模型可能理由写得很好，但判断错误。
- 生成整个 JSON 比输出一个分类分数更慢。
- token-level loss 不一定直接对应相关性准确率。

------

## 28.2 分类头 + 生成头

共享 Transformer backbone：

```
query + item
       ↓
Transformer Backbone
       ├── relevance head
       └── generation head
```

相关性 loss：
$$
L_{\text{rel}}
=
-\left[
y\log p+(1-y)\log(1-p)
\right]
$$
理由生成 loss：
$$
L_{\text{reason}}
=
-\sum_t
\log P(r_t\mid q,d,r_{<t})
$$
总 loss：
$$
L=
\alpha L_{\text{rel}}
+
\beta L_{\text{reason}}
$$
这里的 $\alpha,\beta$ 决定两个任务的梯度强度。

反向传播时：
$$
\nabla_\theta L
=
\alpha\nabla_\theta L_{\text{rel}}
+
\beta\nabla_\theta L_{\text{reason}}
$$
也就是说，共享 backbone 同时收到两种梯度。

可能发生：

- 两种梯度方向一致：互相促进。
- 两种梯度冲突：一个任务提升，另一个下降。
- 某个 loss 数值过大：主导整个训练。

所以多任务训练不仅是“把两个 loss 加起来”，还要监控：

- 两个 loss 的数量级。
- 梯度范数。
- 任务采样比例。
- $\alpha,\beta$ 权重。
- 是否先单任务 warmup。

------

## 28.3 Pairwise 排序 loss

虽然你当前不做最终排序，但如果未来重新训练相关性 score，可以构造正负 pair：
$$
(q,d^+,d^-)
$$
模型输出：
$$
s^+=f(q,d^+)
$$
Pairwise logistic loss：
$$
L_{\text{pair}}
=
\log
\left(
1+\exp(-(s^+-s^-))
\right)
$$
它希望：
$$
s^+>s^-
$$
与普通二分类相比，它直接学习候选之间的相对关系。

------

# 二十九、SFT、GRPO、Rerank 在数学上的统一视角

它们表面上是不同任务，但底层训练链路非常相似：

```
输入
↓
Transformer
↓
logits / score
↓
任务 loss
↓
backward
↓
参数梯度
↓
optimizer.step()
```

区别主要在 loss。

| 任务            | 模型输出              | loss 的监督信号       |
| --------------- | --------------------- | --------------------- |
| SFT             | token logits          | 标准答案 token        |
| GRPO            | token log probability | reward / advantage    |
| Rerank 分类     | relevance score       | 正负标签              |
| Pairwise Rerank | 两个候选分数          | 正候选应高于负候选    |
| 理由生成        | token logits          | 标准理由 token        |
| 多任务 Rerank   | score + token logits  | 分类 loss + 生成 loss |

因此面试时可以说：

> 不同后训练方法的核心差异，往往不是 Transformer 的前向结构发生了根本变化，而是训练样本如何构造、loss 如何定义、梯度如何加权以及哪些参数参与更新。

------

# 三十、一个标准 PyTorch 训练循环

```
model.train()

for batch in train_dataloader:
    optimizer.zero_grad(set_to_none=True)

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    loss = outputs.loss

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    scheduler.step()
```

逐行解释。

------

## `model.train()`

将模型切换到训练模式。

它会影响：

- Dropout
- 某些 normalization 层
- 其他区分 train/eval 的模块

它不会自动开始训练，也不会自动计算梯度。

------

## `optimizer.zero_grad()`

清除上一轮累积梯度。

------

## `outputs = model(...)`

执行前向传播，构建动态计算图。

------

## `loss = outputs.loss`

通常模型内部已经完成：

- label shift
- logits reshape
- Cross Entropy
- ignore index 处理

------

## `loss.backward()`

根据计算图计算梯度，并累积到：

```
param.grad
```

------

## `clip_grad_norm_`

如果梯度整体范数过大，将其缩放到指定范围。

假设所有梯度组成向量 $g$，其范数：
$$
\|g\|_2
$$
如果：
$$
\|g\|_2>M
$$
则缩放：
$$
g\leftarrow g\frac{M}{\|g\|_2}
$$
它主要用于降低梯度爆炸风险。

------

## `optimizer.step()`

根据 optimizer 规则更新参数。

普通 SGD：
$$
\theta\leftarrow\theta-\eta g
$$
AdamW 则会结合：

- 一阶动量。
- 二阶动量。
- 自适应学习率。
- weight decay。

------

## `scheduler.step()`

更新学习率，例如：

- warmup
- cosine decay
- linear decay

------

# 三十一、面试问题一：`loss.backward()` 到底做了什么

## 推荐回答

> 在 PyTorch 中，前向传播会动态构建计算图，图中的每个节点记录对应算子和反向传播规则。调用 `loss.backward()` 时，PyTorch 从标量 loss 开始，将它对自身的梯度设为 1，然后按照逆拓扑顺序遍历计算图。每个算子根据上游梯度和局部导数，通过链式法则计算输入梯度。最终，所有 `requires_grad=True` 的叶子参数会把梯度累积到 `.grad` 中。`backward()` 本身不更新参数，真正的更新由 `optimizer.step()` 完成。

追问时可补充：

- 梯度默认累积。
- 图默认在 backward 后释放。
- 混合精度时可能先经过 loss scaling。
- DDP 中 backward 期间还会触发梯度同步。
- gradient checkpointing 会在 backward 时重新计算部分前向。

------

# 三十二、面试问题二：为什么需要 `optimizer.zero_grad()`

## 推荐回答

> 因为 PyTorch 默认会将新计算的梯度累加到参数现有的 `.grad` 中，而不是覆盖。如果每个训练 step 前不清空梯度，那么当前 batch 的梯度会与之前 batch 的梯度叠加，导致实际更新与预期不一致。因此标准训练循环通常先执行 `optimizer.zero_grad()`，再前向、反向和更新。梯度累积训练是一个例外，它会故意执行多次 backward，达到累积步数后再 step 和 zero grad。

追问：

为什么默认累积？

> 因为一个参数可能在计算图中被多次使用，也可能有多条路径影响 loss，数学上这些梯度本来就应该相加。同时梯度累积也可以用来模拟更大的 batch size。

------

# 三十三、面试问题三：Cross Entropy 为什么适合语言模型

## 推荐回答

> 因果语言模型将每个位置的下一个 token 预测建模为词表上的多分类问题。模型输出词表 logits，通过 Softmax 得到条件概率分布。Cross Entropy 等价于正确 token 的负对数概率，因此最小化所有 token 的 Cross Entropy，就等价于最大化训练序列在模型下的条件似然。它对 logits 可微，并且 Softmax 与 Cross Entropy 结合后的梯度是预测概率减 one-hot 标签，优化形式简单且稳定。

进一步回答局限性：

> 但 token-level Cross Entropy 与序列级业务指标并不完全一致，所以在格式遵循、偏好对齐、长轨迹推理或业务奖励场景中，还会使用 DPO、GRPO 等方法进行后训练。

------

# 三十四、面试问题四：一个 token 的 loss 怎样传回 Transformer 参数

## 推荐回答

> 对位置 $t$，模型通过 LM Head 将最终隐藏状态映射为词表 logits。Cross Entropy 对 logits 的梯度是预测分布减去正确 token 的 one-hot 分布。这个梯度先通过 LM Head 传回位置 $t$ 的最终隐藏状态，然后依次穿过最终 Norm、各层残差连接、MLP 和 Self-Attention。由于 Self-Attention 会聚合当前位置之前的 token 信息，梯度也会传到相关历史位置的表示，并继续传回每一层的 $W_Q、W_K、W_V、W_O$、MLP 参数、Norm 参数和 Embedding。反向传播的数学基础就是计算图上的链式法则。

可以再加一句很有区分度的话：

> 由于 Transformer 各层参数在所有 token 位置共享，所以一个 token 产生的 loss 会对这些共享参数贡献一部分梯度；一个 batch 的最终梯度则是所有有效 token 梯度的聚合。

------

# 三十五、容易被追问的细节

## 35.1 `backward()` 和 `optimizer.step()` 的区别

```
backward：计算梯度
step：使用梯度更新参数
```

没有 `backward()`：

```
optimizer.step()
```

参数没有可用梯度，无法正常学习。

只有 `backward()`，没有 `step()`：

```
loss.backward()
```

梯度算出来了，但参数不改变。

------

## 35.2 `model.eval()` 是否关闭梯度

不会。

```
model.eval()
```

只切换模型行为，例如关闭 Dropout。

真正关闭梯度：

```
with torch.no_grad():
    outputs = model(inputs)
```

或者：

```
with torch.inference_mode():
    outputs = model(inputs)
```

------

## 35.3 为什么 loss 降低不一定意味着业务效果提升

因为优化的是训练 loss：
$$
L_{\text{train}}
$$
但业务关心的可能是：

- Recall
- NDCG
- Pass@K
- 格式正确率
- 推荐点击率
- 相关性准确率
- 理由忠实度

训练 loss 只是代理目标。

如果代理目标和真实业务目标不一致，就可能出现：

```
eval_loss 降低
但平台指标下降
```

这与你之前 LLMRec 和 Rerank 项目中的一些现象直接相关。

------

## 35.4 为什么梯度可能爆炸或消失

链式法则中需要连续相乘：
$$
\frac{\partial L}{\partial x}
=
\prod_{i=1}^{n}
\frac{\partial h_i}{\partial h_{i-1}}
$$
如果每个局部导数都小于 1，连续相乘后趋近 0：
$$
0.5^{100}\approx0
$$
这叫梯度消失。

如果局部导数经常大于 1，连续相乘后可能非常大：
$$
2^{100}
$$
这叫梯度爆炸。

Transformer 通过：

- 残差连接。
- Norm。
- 合理初始化。
- attention scaling。
- 梯度裁剪。
- 学习率 warmup。

改善训练稳定性。

------

# 三十六、今天必须独立手写的内容

不要只运行上面的完整代码。建议重新打开一个空文件，不看答案完成下面四步。

## 第一遍：只写前向传播

```
z1 = ...
a1 = ...
logits = ...
probs = ...
loss = ...
```

要求能说出每个变量的形状。

------

## 第二遍：写反向传播

```
dlogits = ...
dW2 = ...
db2 = ...
da1 = ...
dz1 = ...
dW1 = ...
db1 = ...
```

要求不靠死记硬背，而是用形状检查。

------

## 第三遍：用 PyTorch 验证

```
loss_t.backward()
```

然后比较：

```
np.max(np.abs(manual_grad - torch_grad))
```

------

## 第四遍：写训练循环

连续训练 100 步：

```
for step in range(100):
    # forward
    # backward
    # update

    if step % 10 == 0:
        print(step, loss)
```

观察 loss 是否整体下降。

------

# 三十七、建议额外完成一个梯度检查

PyTorch 验证属于“自动微分对照”。

还可以使用数值梯度检查。

根据导数定义：
$$
\frac{\partial L}{\partial w}
\approx
\frac{L(w+\epsilon)-L(w-\epsilon)}
{2\epsilon}
$$
例如检查 $W_1[0,0]$：

```
epsilon = 1e-5

original_value = W1[0, 0]

W1[0, 0] = original_value + epsilon
loss_plus = compute_loss(X, y, W1, b1, W2, b2)

W1[0, 0] = original_value - epsilon
loss_minus = compute_loss(X, y, W1, b1, W2, b2)

W1[0, 0] = original_value

numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)

print("Numerical gradient:", numerical_grad)
print("Manual gradient:", dW1[0, 0])
```

数值梯度非常慢，但特别适合验证手写 backward。

------

# 三十八、项目映射练习

请你脱离资料，完成下面这张表。

| 原理          | SFT 中对应什么              | GRPO 中对应什么              | Rerank 中对应什么         |
| ------------- | --------------------------- | ---------------------------- | ------------------------- |
| logits        | 每个位置的词表分数          | rollout token 的词表分数     | 相关性分数或生成 logits   |
| Softmax       | 下一个 token 概率           | policy token 概率            | 分类概率或生成概率        |
| Cross Entropy | 标准答案 token loss         | 通常不是唯一目标             | 分类 CE、理由生成 CE      |
| 梯度          | 提高 GT token 概率          | advantage 加权 logprob 梯度  | 提高相关判断和理由质量    |
| backward      | token loss 传回 Transformer | policy loss 传回 Transformer | 多任务 loss 传回 backbone |
| optimizer     | 更新全参或 LoRA             | 更新 actor 参数              | 更新 backbone/head/LoRA   |
| mask          | prompt/ padding 不计 loss   | response mask                | 无效候选或 prompt mask    |
| 梯度累积      | 模拟大 batch                | 累积 rollout micro-batch     | 累积 query-item 样本      |

你真正应该形成的理解是：

> SFT、GRPO 和 Rerank 并不是三套完全割裂的训练系统。它们共享 Transformer 前向、计算图、链式法则和优化器机制，主要差别是数据组织方式和 loss 定义。

------

# 三十九、30 分钟面试自测

每道题先口述 2 分钟，再看答案。

## 基础题

1. 什么是梯度？
2. 链式法则为什么是反向传播的基础？
3. 前向传播和反向传播分别保存、计算什么？
4. 为什么反向传播从 loss 开始？
5. 为什么 Softmax 前通常减去最大 logit？
6. ReLU 的反向传播是什么？
7. 为什么多个计算路径的梯度需要相加？
8. 为什么残差连接有利于训练深层网络？

## PyTorch 题

1. `requires_grad=True` 是什么意思？
2. `.grad` 在什么时候产生？
3. 为什么 `.grad` 默认累积？
4. `backward()` 会不会更新参数？
5. `optimizer.step()` 使用什么信息？
6. `model.eval()` 是否等价于 `torch.no_grad()`？
7. 为什么 gradient checkpointing 能节省显存？
8. 为什么 mixed precision 需要 GradScaler？

## LLM 题

1. 因果语言模型的 labels 为什么需要 shift？
2. prompt token 为什么常设置为 `-100`？
3. 一个 token 的 CE 梯度是什么？
4. token loss 如何传到 Attention 参数？
5. 为什么当前 token 的 loss 会影响历史 token 表示？
6. 为什么不会通过 causal attention 影响未来 token？
7. SFT loss 下降为什么不一定代表生成质量提升？
8. GRPO 的 reward 为什么可以不可微？

## 项目题

1. LoRA 冻结基础参数后，梯度还会经过基础模型吗？
2. gradient accumulation 与扩大 batch size 有什么关系？
3. packing 为什么可能改变训练效果？
4. 多任务 Rerank 中两个 loss 冲突怎么办？
5. 分类 loss 和理由生成 loss 应该如何加权？
6. GRPO 中 advantage 为负意味着什么？
7. KL 系数过大或过小有什么影响？
8. 为什么平台指标可能和本地 eval loss 不一致？

------

# 四十、最后 10 分钟复盘模板

建议今天直接按下面格式记录。

## 1. 今天真正理解的内容

示例：

> 我理解了反向传播并不是一个独立于导数的新算法，而是从 loss 开始，沿计算图反向重复应用链式法则。每个节点收到上游梯度，再乘以自己的局部梯度，将结果传给前面的节点。

## 2. 今天能够独立推导的公式

必须至少写出：
$$
\frac{\partial L}{\partial z}
=
p-\operatorname{onehot}(y)
$$
以及两层网络：
$$
dW_2=A_1^\top dZ_2
$$

## 3. 仍然模糊的地方

不要写“反向传播不太懂”这种宽泛内容，要具体，例如：

> 我暂时还不能独立推导 LayerNorm 的反向传播。

> 我知道 Attention 梯度会传到 Q、K、V，但暂时无法写出 Softmax Attention 的完整矩阵梯度。

> 我还没有完全理解梯度累积时 loss 是否必须除以 accumulation steps。

## 4. 明天要完成的一个行动

示例：

> 明天不看资料，重新手写两层网络，并加入数值梯度检查；然后用 PyTorch 打印每一层参数的梯度范数。

------

# 今天的验收标准

今天不要以“看完了”为完成标准，而要满足下面五个条件：

1. 能从头解释链式法则和反向传播。
2. 能推导 Softmax + Cross Entropy 的梯度。
3. 能独立手写两层网络的 forward 和 backward。
4. 能解释 `backward()`、`zero_grad()` 和 `step()`。
5. 能把 token loss 的梯度路径映射到 SFT、GRPO 和 Rerank。

最核心的一句话是：

> 后训练方法可以改变数据、loss 和梯度权重，但最终都要把一个标量 loss，通过链式法则传回 Transformer 的可训练参数。