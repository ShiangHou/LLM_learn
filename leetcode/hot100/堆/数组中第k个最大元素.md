# 数组中第k个最大元素

## 堆

这道题说找到第k大

那我们弄一个堆，


## 快排

快排的思路是，随机选择一个基准，然后把比基准小的放在左边，比基准大的放在右边，，然后递归左右就行

看左程云的快排

思路就是，最后不是输出一个下标嘛，即代表着左边都是比自己小的，右边都是比自己大的

那么这个下标不就正好是 ”第几个小“的嘛

所以说，思路是这样的

先随机选择一个基准，然后找到


如果这个基准的位置正好是”第K大的“，那么就返回，如果不是，看一下下标，如果比第K大还大，那么就在左边的数组，如果是比第k大还小，那么就是右边的数组



```python 
def partition(l,r):
    #随机选择一个x
    x_i = random.randint(l,r)#注意这里是下标了,随机选一个
    x = nums[x_i]#存下这个大小
    #放到最左边
    x_i = 0
    a = l
    for i in range(l,r+1):
        if nums[i] <= x:#小，丢在左边
            nums[i],nums[a] = nums[a],nums[i]
            if nums[a] == x:
                x_i = a
            a += 1
    #最后锚点
    nums[a-1],nums[x_i] = nums[x_i],nums[a-1]
    return a-1
```
不行，这个不能用推土机了，因为有骚猪样例全都是一样的，导致报错

```python 
def partition(left, right):
            # 引入随机化，避免极端测试用例（比如原本就有序的数组）导致超时
            random_index = random.randint(left, right)
            nums[left], nums[random_index] = nums[random_index], nums[left]
            
            pivot = nums[left]
            l = left
            r = right
            
            while l < r:#意思就是，找到右边第一个比基准小的，以及左边第一个比基准大的，交换，
                while l < r and nums[r] >= pivot:
                    r -= 1
                while l < r and nums[l] <= pivot:
                    l += 1
                if l < r:
                    nums[l], nums[r] = nums[r], nums[l]
            
            # 基准归位
            nums[left], nums[l] = nums[l], nums[left]
            return l  # 返回基准值最终的绝对位置
```

妈的还是不行，报错，因为有很多重复的

要用荷兰国旗的那种写法

```python
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        def quick(nums,k):
            provit = random.choice(nums)
            big = [x for x in nums if x> provit]
            equal = [x for x in nums if x == provit]
            small = [x for x in nums if x < provit]
            if k <= len(big):#说明在后面,即big里面
                return quick(big,k)
            if k > len(big)+len(equal):
                return quick(small,k-len(big)-len(equal))
            return provit
        return quick(nums,k)
```
背诵就完了
