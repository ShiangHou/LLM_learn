# 合并K个升序链表

## 初步思路

这道题可以用小顶堆去做

所谓的小顶堆，就是说，我的根节点是最小的

那么这里就是，首先我有k个链表嘛，然后我就构建k个数量的小顶对

我把k个头都放入小顶对中，小顶堆的root就是最小的，然后我从堆中弹出一个，接上去，如果弹出的next还有，就也放入到堆中

这里就是用所谓的优先级队列，即import heapq

```python 
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        #准备一个优先级队列
        import heapq
        heap = []
        for i,node in enumerate(lists):
            if node:
                heapq.heappush(heap,(node.val,i,node))#插入i是为了防止值相等的时候报错
        dmmy = ListNode(0)#先准备一个虚拟头节点
        cur = dmmy
        while heap:
            #取出一个
            val,i,node = heapq.heappop(heap)
            cur.next = node
            cur = cur.next

            #把node的下一个也放入堆
            if node.next:
                heapq.heappush(heap,(node.next.val,i,node.next))
        return dmmy.next
```