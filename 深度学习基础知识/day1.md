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

# 十一、手写最小两层神经网络

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
exp_logits = np.exp(shifted_logits)
probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# Cross Entropy
correct_probs = probs[np.arange(batch_size), y]
loss = -np.mean(np.log(correct_probs))

print("Manual loss:", loss)


# ============================================================
# 4. 反向传播
# ============================================================

# Softmax + Cross Entropy 的梯度：
# dlogits = (probs - one_hot(y)) / batch_size

dlogits = probs.copy()
dlogits[np.arange(batch_size), y] -= 1.0
dlogits /= batch_size

# 第二层梯度
dW2 = a1.T @ dlogits
db2 = np.sum(dlogits, axis=0)

# 梯度传回隐藏层
da1 = dlogits @ W2.T

# ReLU 反向传播
dz1 = da1 * (z1 > 0)

# 第一层梯度
dW1 = X.T @ dz1
db1 = np.sum(dz1, axis=0)


# ============================================================
# 5. 使用 PyTorch 验证梯度
# ============================================================

# 使用 float64，减少数值误差
X_t = torch.tensor(X, dtype=torch.float64)
y_t = torch.tensor(y, dtype=torch.long)

W1_t = torch.tensor(W1, dtype=torch.float64, requires_grad=True)
b1_t = torch.tensor(b1, dtype=torch.float64, requires_grad=True)

W2_t = torch.tensor(W2, dtype=torch.float64, requires_grad=True)
b2_t = torch.tensor(b2, dtype=torch.float64, requires_grad=True)

z1_t = X_t @ W1_t + b1_t
a1_t = torch.relu(z1_t)
logits_t = a1_t @ W2_t + b2_t

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

# 十二、逐行理解这段反向传播

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

# 十三、PyTorch 的自动微分做了什么

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

# 十四、`loss.backward()` 到底做了什么

面试中可以这样回答：

> `loss.backward()` 会从 loss 节点出发，沿前向过程中构建的动态计算图反向遍历。对每个算子，它利用该算子的局部导数和上游梯度，通过链式法则计算输入梯度。对于 `requires_grad=True` 的叶子张量，例如模型参数，最终梯度会累积到参数的 `.grad` 属性中。它只负责计算和累积梯度，并不会更新参数，参数更新由 `optimizer.step()` 完成。

可以继续补充四个细节。

## 14.1 loss 通常必须是标量

如果 loss 是标量，PyTorch 默认：
$$
\frac{\partial L}{\partial L}=1
$$
如果输出不是标量，需要显式提供上游梯度：

```
y.backward(gradient=torch.ones_like(y))
```

------

## 14.2 梯度存入 `.grad`

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

## 14.3 梯度默认累积

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

## 14.4 计算图通常在 backward 后释放

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

# 十五、为什么需要 `optimizer.zero_grad()`

标准训练循环：

```
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

## 15.1 为什么 PyTorch 要设计成累积，而不是自动覆盖

因为梯度累积本身是一项重要功能。

例如显存只能放一个 micro-batch，但希望有效 batch size 为 8：

```
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

## 15.2 `zero_grad(set_to_none=True)`

常见写法：

```
optimizer.zero_grad(set_to_none=True)
```

它不是把梯度 tensor 全部填成 0，而是把 `.grad` 设置为 `None`。

通常可以减少内存写入，性能略好，也是现代 PyTorch 常用设置。