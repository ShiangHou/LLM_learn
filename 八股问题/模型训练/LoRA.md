# LoRA
关于Lora，你应该知道的一切

LoRA，即Low rank，是为了解决大模型训练时显存的一种手段


什么意思，在大模型微调的时候，对于一个矩阵，人们发现（主要是LoRA的作者发现），对于微调修改后的那个矩阵其实是低rank的，即我们修改的其实没多少

因此，我们可以

$$W = W_0 + \Delta W = W_0 + BA$$

即微调后的 等于 原来的+微调改动的，因为微调改动的是一个低rank的，我们可以弄成两个向量，

两个向量，B和A，B的维度是什么，

想知道维度，乘起来之后肯定是和W一样的维度啊

假设W的维度是d*k，那么很显然，B的维度是d*r，然后A的维度是r*k

此时我们设置A是高斯，然后B是全0就可以

注意，虽然是BA，但是其实最后用的时候是BAX，X是先和A乘，然后在和B乘，x的维度是d，那么A的维度就是d*r，这个就是高斯

此外，Lora还有一个参数是a

## LoRA应用在哪一层

一般来说，原论文里面说可以用到Wq和Wv效果比较好

当然可以用在全部的all liner ，即Wq k v 以及ffn层



## 手撕Lora
```python 
# ✅ SOLUTION

class LoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        
        # Base weight
        self.W = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.W)
        
        if rank > 0:
            # LoRA matrices
            self.A = nn.Parameter(torch.empty(rank, in_features))
            self.B = nn.Parameter(torch.zeros(out_features, rank))
            nn.init.kaiming_uniform_(self.A)
        else:
            self.A = nn.Parameter(torch.empty(0, in_features), requires_grad=False)
            self.B = nn.Parameter(torch.empty(out_features, 0), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output
        out = x @ self.W.T
        
        # LoRA addition
        if self.rank > 0:
            out = out + (x @ self.A.T @ self.B.T) * (self.alpha / self.rank)
        
        return out


```


