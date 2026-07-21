# K个一组反转链表
给你链表的头节点 head ，每 k 个节点一组进行翻转，请你返回修改后的链表。

k 是一个正整数，它的值小于或等于链表的长度。如果节点总数不是 k 的整数倍，那么请将最后剩余的节点保持原有顺序。

你不能只是单纯的改变节点内部的值，而是需要实际进行节点交换。

## 初步思路
首先准备两个，一个是给定一个node，拿到第k个的尾巴

一个是给定start和end，反转链表，返回反转后的start，

准备几个变量
start和end是需要反转的两端，pre和_next是反转时候的前一个和后一个

逻辑是，先往后走k个，然后如果没走到，即end是NOne，那么就不反转

走到了，执行反转，然后更新

```python 
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        
        #准备两个函数
    
        def team_k(node ,k)->ListNode:#负责找到k个
            #走k-1下就行
            while k-1 !=0 and node:
                node = node.next
                k -=1
            return node
        def swap(start,end)->ListNode:#反转从start到end
            next_group_start = end.next#存一下下一组的
            end.next = None#断开
            #标准的反转
            pre = None
            cur = start
            while cur :
                _next = cur.next
                cur.next = pre
                pre = cur
                cur = _next
            #pre是反转后的头，start是最后面的，直接接上
            start.next = next_group_start
            return start#即下一组的前一个，pre
        
        #主流程
        pre = None
        start = head
        end = team_k(start ,k)
        if not end:
            return head
        #特殊处理第一组，拿到最后ans的头,即end
        head = end#记一下最后的答案的头
        pre = swap(start,end)#反转，拿到钱一个
        #循环
        while pre.next:
            start = pre.next
            end = team_k(start ,k)
            if not end:
                return head
            pre.next = end#直接先接住
            pre = swap(start,end)#另一个是在这里面接到next_group
        return head





```