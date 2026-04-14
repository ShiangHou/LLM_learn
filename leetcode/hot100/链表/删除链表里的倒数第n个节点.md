# 删除链表里面的倒数第n个节点

要求只遍历一遍，那么用双指针，

道理很好懂，首先问的是倒数第n个节点，那么我只要让fast先走n步，然后fast和slow再一起走，这样fast和slow就只差了n个，然后直接把slow.next = slow.next.next就行了

```python 
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        slow,fast = head,head 
        for i in range(n):
            fast = fast.next
        if fast == None:#如果删除的是头节点,即倒数第N个
            return head.next
        while fast.next:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return head

        # 1 2 3 4 5
        #     s   f


```