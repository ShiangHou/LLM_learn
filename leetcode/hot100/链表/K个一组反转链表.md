# K个一组反转链表

## 初步思路
这里其实比较好想

思路是，找到K个，然后反转，然后再往下找K个，如果不足K个就直接不反转

需要一点小心思，我们准备几个变量

一个head，用来存储最后的返回
pre 即需要反转的链表的前一个节点
start 需要反转的开始节点
end 需要反转的最后一个节点
_next 下一波的头节点

首先我们先准备一个反转链表的函数，入参是头节点，出参是反转好的头节点

然后主流程

开始的时候，我们让end往后走，走k个，如果发现走到了None，那就不反转，说明搞完了，直接返回head

如果走到了第k个，那么需要将start开始，到end进行反转，

在调用反转函数之前，记得吧end给指到空，（记得存一下end.next,这是下一个start的开始）

完成后，让pre指向新链表的头，然后pre这个转向新链表的尾巴，把start转为原来我们存的next

然后继续让end往前冲就行了，冲到None直接结束循环

```python 
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        pre = None #需要反转的前一个节点
        start = head #需要反转的第一个节点
        end = self.team_k(start,k)#需要反转的最后一个节点
        #特殊处理一下第一组吧
        if not end:
            return head
        head = end#第一组换头
        pre = self.swap(start,end)#直接反转，输出的是反转后的最后一个，那么就是下一组的第一个，也就是pre

        while pre.next:
            start = pre.next#确定下一个反转的头
            end = self.team_k(start,k)#下一个反转的尾巴
            if not end:
                return head
            pre.next = end
            pre = self.swap(start,end)
        return head
        



                    
    def team_k(self,node,k):#输入一个链表头，返回第k个的尾巴，如果没有，返回None 
        while k-1 != 0 and node:
            node = node.next
            k -= 1
        return node
        
    def swap(self,start,end):
        #反转链表,输出新链表的最后一个，也就是下一个的pre
        next_group_start = end.next#先存一下
        end.next = None
        pre = None
        cur = start
        _next = None
        while cur:
            _next = cur.next
            cur.next = pre
            pre = cur
            cur = _next
        #现在start是在最后面了
        start.next = next_group_start
        return start#输出尾巴

```