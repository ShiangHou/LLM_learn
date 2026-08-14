# MHA



```python 
import torch.nn as nn
import torch

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,n_head):
        super().__init__()


        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model/n_head

        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.W_o = nn.Linear(d_model,d_model)

    def forward(self,q,k,v):
        B,S_q,_ = q.size()
        B,S_k,_ = k.size()
        Q = self.W_q(q).view(B,S_q,self.n_head,self.head_dim).transpose(1,2)
        K = self.W_k(k).view(B,S_k,self.n_head,self.head_dim).transpose(1,2)
        V = self.W_v(v).view(B,S_k,self.n_head,self.head_dim).transpose(1,2)

        #计算score
        score = Q @K.transpose(2,3) // math.sqrt(self.head_dim)

        score = torch.softmax(score,dim = -1)@V
        score = score.transpose(1,2).contiguous().view(B,S_q,self.d_model)

        output = self.W_o(score)

        return output




```

# 改成GQA

把MHA改成GQA

就是把q的头分组，分到更少的kv cache里面



既然如此，记得定义q的head和kv的head就行

```python 
import torch

class GroupQueryAttention(nn.Module):
    def __init__(self,d_model,q_heads,kv_heads):
        __init__().super()
        assert d_model % q_heads = 0
        assert q_heads % kv_heads = 0

        self.d_model = d_model
        self.q_heads = q_heads
        self.kv_head = kv_heads

        self.hiden_dim = d_model // q_heads
        self.group = q_heads // kv_heads

        self.W_q = nn.Linear(self.d_model,self.q_heads*self.hiden_dim  )
        self.W_k = nn.Linear(self.d_model,self.kv_heads*self.hiden_dim  )
        self.W_v = nn.Linear(self.d_model,self.kv_heads*self.hiden_dim  )  

        self.W_o = nn.Linear(self.d_model,d_model)

    def forward(self,q,k,v,attention_mask = None):
        B,S_q,_ = q.size()
        _,S_k,_ = k.size()

        Q = self.W_q(q).view(B,S_q,self.q_heads,self.head_dim)
        K = self.W_k(k).view(B,S_k,self.kv_heads,self.head_dim)
        V = self.W_v(v).view(B,S_k,self.kv_heads,self.head_dim)

        #把每个kv复制一下
        K = K.repeat_interleave(self.num_groups,dim=1)
        V = V.repeat_interleave(self.num_groups,dim=1)

        score = Q@K.transpose(1,2)

        score = score/math.sqrt(self.head_dim)

        if attention_mask:
            score.mask_fill(attention_mask == 0,torch.finfo(score.dtype).min)
        attention_weight = torch.softmax(score,dim = -1)
        output = attention_weight@V
        return self.W_o(output)

  
  
```

