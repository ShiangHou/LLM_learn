# 大模型外推

https://zhuanlan.zhihu.com/p/25241219397

参考这个

关于大模型外推，一般主要是解决两个问题



一个是 位置编码能不能外推，这个问题是在说，比如我训练的时候的最大是 100的长度，那么在使用推理的时候如果是200的长度，那么旋转位置编码能不能自己往外推出去，所以这个跟ROPE和YARN等有关



另一个问题是context length，就是现在大家在说的1 million的上下文，如扩展上下文，这种技术以及超出了，涉及到模型框架、原生长上下文的训练，等等，



比如qwen3 用的还是YARN，但是现在的Kimi3 deepseek v4、GLM等都是各种，什么DSA、HCA、KDA等等比较fancy的模型框架，外加超长上下文训练，这个需要沉淀一下技术报告才行



先说怎么骗ROPE达到外推的吧



## 位置编码类



先说一下ROPE，先把之前的笔记copy过来

### 旋转位置编码RoPE

旋转位置编码，是在计算完q和k之后进行的，即
$$
\begin{align*}
x_m' &= W_q x_m e^{i m \theta} = (W_q x_m)e^{i m \theta} = q_m e^{i m \theta} \\
x_n' &= W_k x_n e^{i n \theta} = (W_k x_n)e^{i n \theta} = k_n e^{i n \theta}
\end{align*}
$$

此处的m和n表示两个位置，可以发现，$\boldsymbol{W}_q$即计算q的参数矩阵，k也同理。$\theta$和attention论文里的是同一个.

在两个位置相乘后有如下结果
$$
x_m^T x_n' = (q_m^1 \, q_m^2) \begin{pmatrix} \cos((m - n)\theta) & -\sin((m - n)\theta) \\ \sin((m - n)\theta) & \cos((m - n)\theta) \end{pmatrix} \begin{pmatrix} k_n^1 \\ k_n^2 \end{pmatrix}
$$

证明
首先，有
$$
q_m = \begin{pmatrix} W_q^{11} & W_q^{12} \\ W_q^{21} & W_q^{22} \end{pmatrix} \begin{pmatrix} x_m^1 \\ x_m^2 \end{pmatrix} = \begin{pmatrix} q_m^1 \\ q_m^2 \end{pmatrix}
$$

二维向量可以表达成一个实数坐标轴和虚数坐标轴
$$
q_m = q_m^1 + i q_m^2
$$

而根据欧拉公式，
$$
e^{im\theta} = \cos(m\theta) + i\sin(m\theta)
$$
因此，乘开，有
$$
q_m e^{im\theta} = (q_m^1 \cos(m\theta) - q_m^2 \sin(m\theta)) + i(q_m^2 \cos(m\theta) + q_m^1 \sin(m\theta))
$$

转化为实部和虚部的向量形式，有
$$
\left[ q_m^1 \cos(m\theta) - q_m^2 \sin(m\theta) , q_m^2 \cos(m\theta) - q_m^1 \sin(m\theta) \right] =\begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_m^1 \\ q_m^2 \end{pmatrix} 
$$
所以对于位置m的q的编码位置后的向量，有
$$
\begin{align*}
x_m' &= W_q x_m e^{i m \theta} = (W_q x_m)e^{i m \theta} = q_m e^{i m \theta} \\
&= \begin{pmatrix} \cos(m\theta) & -\sin(m\theta) \\ \sin(m\theta) & \cos(m\theta) \end{pmatrix} \begin{pmatrix} q_m^1 \\ q_m^2 \end{pmatrix}
\end{align*}
$$
对于n位置的k，也有
$$
\begin{align*}
x_n' &= W_k x_n e^{in\theta} = (W_k x_n)e^{in\theta} = q_k e^{in\theta} \\
&= \begin{pmatrix} \cos(n\theta) & -\sin(n\theta) \\ \sin(n\theta) & \cos(n\theta) \end{pmatrix} \begin{pmatrix} k_n^1 \\ k_n^2 \end{pmatrix}
\end{align*}
$$
相乘，就会得到最终的结果
$$
x_m'^Tx_n'= (q_m^1\ q_m^2) \begin{pmatrix} \cos((m - n)\theta) & -\sin((m - n)\theta) \\ \sin((m - n)\theta) & \cos((m - n)\theta) \end{pmatrix} \begin{pmatrix} k_n^1 \\ k_n^2 \end{pmatrix}
$$
还有一种方法是这样的证明
$$
g(x_m', x_n', m - n) = \text{Re}\left[ (W_q x_m)(W_k x_n)^* e^{i(m - n)\theta} \right]
$$

其中Re表示取实数部分，*表示共轭
扩展到多维，把每两个进行旋转即可
$$
\begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2 -1} & -\sin m\theta_{d/2 -1} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2 -1} & \cos m\theta_{d/2 -1}
\end{pmatrix}
\begin{pmatrix}
q_0 \\
q_1 \\
q_2 \\
q_3 \\
\vdots \\
q_{d-2} \\
q_{d-1}
\end{pmatrix}
$$

矩阵过于稀疏，可以简化为
$$
\begin{pmatrix} q_0 \\ q_1 \\ q_2 \\ q_3 \\ \vdots \\ q_{d - 2} \\ q_{d - 1} \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_0 \\ \cos m\theta_0 \\ \cos m\theta_1 \\ \cos m\theta_1 \\ \vdots \\ \cos m\theta_{d / 2 - 1} \\ \cos m\theta_{d / 2 - 1} \end{pmatrix} + \begin{pmatrix} -q_1 \\ q_0 \\ -q_3 \\ q_2 \\ \vdots \\ -q_{d - 1} \\ q_{d - 2} \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_0 \\ \sin m\theta_0 \\ \sin m\theta_1 \\ \sin m\theta_1 \\ \vdots \\ \sin m\theta_{d / 2 - 1} \\ \sin m\theta_{d / 2 - 1} \end{pmatrix}
$$



总结一下之前的笔记，ROPE需要搞清楚两个概念，分别是位置和维度

对于一个$m\theta_{i}$，这里的m指的是位置，即在一个sequence中的位置，而$\theta_{i}$指的是这一个token的第i个维度

如果是第0个位置，那么就一直是0啊，因为m是0，如果是第一个维度，那么m就是1，等等

然后对于这一个token上的不同的维度，我们是两两在一起，然后旋转，维度越大旋转越慢，目的是，如果旋转都一样的话，那么转360后就还是0了，就学不到了，这样，然后讲故事的话，低纬度的学的是局部信息，然后高纬度的学的是全局信息，这样子。





问题的提出

比如说，我在训练的时候用的是32K的，假如是有些地方只旋转了90度，但是在用的时候，给我填了128K，那么可能有的旋转了270度

从纯数学的角度上来说，自然可以“算”出来，但是模型由于在训练的时候没见过，



更麻烦的是，有些低频维度在训练区间内连一圈都没有转完，因此模型可能把这一段近似单调的旋转轨迹偷偷当成某种**绝对位置提示**。一旦推理超过训练长度，这种规律就失效。

PI

PI就是position Interpolation，即位置插入

这个方法是说，比如我训练的时候是32K，但是最后见到的是128K

那么，我直接把128K压缩到32K不就行了，即我把那个m除4就ok了

这样的话，比如真实的位置是100，那么经过PI后，ROPE的那个m就是25，这样就把128K压缩压回到了32K见过的位置，

但是问题就是说，把分辨率给降低了，即所有的维度都降低了

而且模型对高频的非常的敏感



更好的方法是，不同的维度降低的不一样

统一缩放会损失高频位置的信息，因此更合理的办法是**不同频率采用不同程度的缩放**

 NTK-aware Scaling

这里的提出就是说，我缩放，对于维度比较高，旋转比较少的地方，32K可能一圈都没走完，那么再压缩就更少了，会有问题，所以思想很简单，就是我的维度低的地方，高频的地方可以多多的缩放，低频维度高的地方少少的缩放，即高频少压，低频多压（压theta相当于外推m）

NTK by parts

给了一个a和b，要求小于a的不压，大于a的全压

YaRN

就是 NTK +attentio scaling







总结一下就是

```	​
RoPE
φ(m,d) = m θd
│
│
├── PI
│
│   θd' = θd / s
│
│   所有 frequency 一起压
│
│   优点：全部回到训练范围
│   缺点：损失高频/local position信息
│
▼
NTK-aware
│
│   θd' = θd · s^(-2d/(D-2))
│
│   高频少压
│   低频多压
│
│   优点：保留高频信息
│   缺点：部分dimension会extrapolate
│
▼
NTK-by-parts
│
│   r(d) = L / λd
│
│   高频：r > β → 不压
│   低频：r < α → θ/s
│   中频：平滑混合
│
│   关键：targeted interpolation
│
▼
YaRN
    │
    ├── NTK-by-parts
    │
    └── Attention scaling
        softmax(QKᵀ / t√D)

```

RoPE 对第 \(m\) 个 token、某一组 embedding 维度 \(d\) 的旋转角度为：

$$
\phi_{m,d}=m\theta_d
$$

其中 \(m\) 是 token 的位置，\(\theta_d\) 是这一组维度对应的旋转频率。  
当模型训练长度只有 \(L\)，但推理长度扩到 \(sL\) 时，新的 \(m\) 会让旋转角度进入训练时没见过的范围，因此需要做长度外推。

## 

### PI：所有频率统一压缩

Position Interpolation 直接把位置缩小：

$$
m'=\frac{m}{s}
$$

等价于：

$$
\theta_d'=\frac{\theta_d}{s}
$$

这样 \(sL\) 的位置会被压回原来的 \(L\) 范围，避免位置编码超出训练分布。

**问题：**所有频率都被统一缩小，高频维度也被压缩，导致相邻 token 的角度差变小，局部位置分辨率下降。

---

### NTK-aware：高频少缩，低频多缩

NTK-aware 不再让所有 \(\theta_d\) 除以同一个 \(s\)，而是根据维度使用不同的缩放比例：

$$
\theta_d'
=
\theta_d\cdot s^{-\frac{2d}{D-2}}
$$

效果是：

- 高频维度几乎不变，保留局部位置分辨率；
- 低频维度接近 PI，负责把长距离位置拉回训练范围；
- 中间频率平滑过渡。

**为什么这样做：**高频维度在训练窗口内通常已经旋转过很多圈，继续外推风险较小；低频维度可能连一圈都没走完，直接外推更容易 OOD。

**问题：**它只是按照维度连续缩放，没有直接判断某个频率在训练阶段到底“转过多少圈”，因此仍然比较粗糙。

---

### NTK-by-parts：按“训练时转过多少圈”决定缩多少

先定义某个频率的波长：

$$
\lambda_d=\frac{2\pi}{\theta_d}
$$

再计算它在训练长度 \(L\) 内转过多少圈：

$$
r(d)=\frac{L}{\lambda_d}
$$

然后分情况处理：

- \(r\) 很大：训练时已经转过很多圈，**不缩放**；
- \(r\) 很小：训练时连一圈都没走完，**完整做 PI**；
- 中间区域：在原频率和 PI 频率之间平滑插值。

可以写成：

$$
\theta_d'
=
(1-\gamma)\frac{\theta_d}{s}
+\gamma\theta_d
$$

其中 \(\gamma\) 根据 \(r(d)\) 决定。

核心思想是：

> 不再单纯看“这是第几维”，而是看“这只 RoPE 的钟在训练阶段到底转过多少圈”。

因此它比 NTK-aware 更有针对性。

---

### YaRN：NTK-by-parts + Attention Scaling

YaRN 保留 NTK-by-parts 的频率处理，同时发现：

> 长上下文不仅会让 RoPE 分布变化，Attention 的 logits / softmax 分布也会发生变化。

因此进一步对 Attention 做 temperature / magnitude scaling：

$$
\operatorname{Softmax}
\left(
\frac{QK^\top}{t\sqrt{d}}
\right)
$$

等价地，也可以对 \(Q,K\) 做相应尺度调整。

所以可以直接记成：

$$
\boxed{
\text{YaRN}
=
\text{NTK-by-parts}
+
\text{Attention Scaling}
}
$$

---

### 一句话总结

$$
\boxed{
\text{PI：所有频率一起压}
\rightarrow
\text{NTK-aware：高频少压、低频多压}
\rightarrow
\text{NTK-by-parts：按训练时转过多少圈有选择地压}
\rightarrow
\text{YaRN：再补上 Attention Scaling}
}
$$

整条演进都在解决同一个矛盾：

> **既要避免长位置进入训练分布之外，又要尽量保留原本的局部位置分辨率。**



## 框架+原生训练类

