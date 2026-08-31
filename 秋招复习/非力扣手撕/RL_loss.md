# RL loss 手撕
## policy gradint 

就是直接advantage乘log就行，

```python 
def pg_loss(advantage,log_prob):
    return -(advantage*log_prob).mean()

```

## PPO

PPO就是多了重要性采样，以及clip的过程

对于重要性采样，我们用torch.exp(log-log)就行
对于clip，我们用torch的clamp方法

```python 
def ppo_loss(advantage,logp,old_logp,eps = 0.2):
    ratio = torch.exp(logp-old_logp)
    sur1 = ratio*advantage
    sur2 = torch.clamp(
        ratio,
        1-eps,
        1+eps
    )*advantage

    loss = -torch.min(sur1,sur2).mean()
    return loss
```


## GRPO
grpo和ppo唯一的区别就是advantage的来源不一样，是用reward算好的均值和方差再去弄的，
reward是一个[]的，
```python 

def grpo_loss(reward,logp,old_logp,eps = 0.2)
    #先把advantage算出来
    #reward :[B,G]
    mean = reward.mean(dim = 1,keepdim = True)
    std = reward.std(dim = 1,keepdim = True,unbiased = False)

    advantage =( reward - mean)/(std+1e-8)
    #[B, G]
    advantage = advantage.unsqueeze(-1)
    #[B, G,1]
    ratio = torch.exp(logp-old_logp)
    sur1 = ratio*advantage
    sur2 = torch.clamp(ratio,1-eps,1+eps,)*advantage

    loss = -torch.min(sur1,sur2).mean()
    return loss
```

## GSPO

GSPO就是重要性采样上进行了遍变化，依旧是用exp加上t就行

```python 
def gspo_loss(reward,logp,old_logp,eps = 0.2):
    #还是先算advantage
    mean = reward.mean(dim = 1,keepdim = True)
    std = reward.std(dim = 1,keepdim = True,unbiased = False)

    advantage = (reward-mean) / (std+1e-8)

    seq_ratio =  (logp - old_logp).mean(dim=-1)#相当于求和完再除，就是求均值
    sur1 = seq_ratio*advantage
    sur2 = torch.clamp(seq_ratio,1-eps,1+eps)* advantage
    loss = -torch.min(sur1,sur2).mean()
    return loss
```

## DPO

四个logb，一个chose，一个reject，然后各跟着了，还有一个beta，然后写就行
```python 
def DPO_LOSS(chosen_log,chosen_reference,reject_log,reject_reference,beta):
    pi_log = chosen_log-reject_log
    ref_log = chosen_reference - reject_reference
    logits_ratio = beta*(pi_log-ref_log)

    loss = -F.logsigmoid(logits_ratio).mean()

    return loss 

```

## KL

```python 
def KL(p_log,q_log):
    p = torch.exp(p_log)

    kl = p*(p_log-q_log)

    return kl.sum(dim = -1).mean()


```