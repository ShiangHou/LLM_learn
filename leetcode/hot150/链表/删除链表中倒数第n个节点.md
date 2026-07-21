# 删除链表中倒数第n个节点


提示
给你一个链表，删除链表的倒数第 n 个结点，并且返回链表的头结点。

head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]
示例 2：

输入：head = [1], n = 1
输出：[]


## 初步思路
用两个指针，一快一慢，相隔n个，如果快的到空了，那么慢的正好就是倒数第n位置，然后直接next=.next.next就行

特殊看一下n为长度的时候，即慢的还是head，此时直接返回head.next就行
想一下，倒数第n个

1 2 3 4 5

比如倒数第2个，就是4

那么f走到5，s应该是到3
f是3，s是1
所以俩人从头开始，f先走k+1步，然后再一起走

```python 
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        if not head:
            return None
        ans = head#记录一下答案
        f,s = head,head
        while n>0:
            f = f.next
            if not f:#说明到头了，返回的倒数n个覆盖了head
                return head.next
            n-=1
        #然后俩一起走
        while f and f.next:
            f = f.next
            s = s.next
        #记录答案
        s.next = s.next.next
        return ans 
        
        

```
