# H指数

给你一个整数数组 citations ，其中 citations[i] 表示研究者的第 i 篇论文被引用的次数。计算并返回该研究者的 h 指数。

根据维基百科上 h 指数的定义：h 代表“高引用次数” ，一名科研人员的 h 指数 是指他（她）至少发表了 h 篇论文，并且 至少 有 h 篇论文被引用次数大于等于 h 。如果 h 有多种可能的值，h 指数 是其中最大的那个。


理解一下题目，就是说，大于等于的是4

这个

```python 


class Solution:
    def hIndex(self, citations: List[int]) -> int:
        #至少有一篇大于等于1，有2篇大于等于2，有三篇大于等于3，没有四篇大于等于4
        #对于[1,3,1],至少有1篇大于等于1，没有2篇大于等于2
        #先遍历一遍，统计出引用次数i的有几篇
        #然后从后往前数就行，累加一个值sum，直到数到sum>=i，就表示有sum个引用次数大于等于i，即找到了这个
        n = len(citations)
        count = [0]*(n+1)#用一个数组模拟一下hash
        for cit in citations:
            count[min(cit,n)] +=1#统计一下，这里是，如果超过了n次，其实直接统计成n次也可也
        s = 0
        #开始数
        for i in range(n,-1,-1):
            s += count[i]
            if s >=i:
                return i
        

```