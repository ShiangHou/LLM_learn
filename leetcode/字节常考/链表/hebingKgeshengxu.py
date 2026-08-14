from typing import Optional,Any


# k = int(input())
# nums_list = []
# for _ in range(k):
#     nums_list.append(list(map(int,input().split())))

nums_list = [[1,1,5],[1,3,4],[2,6]]

class Node:
    def __init__(self,val = 0,next = None):
        self.val = val
        self.next = next


    # def __lt__(self, other):
    #     return self.val < other.val


def build_node(nums:list) -> Node:
    tail = Node(nums[-1])
    cur = tail
    for i in range(len(nums)-2,-1,-1):
        temp = Node(val = nums[i],next = cur)
        cur = temp 
    return cur 
#先把nums的都变成node进去
lists :list[Optional[Node]] = []
for num in nums_list:
    #先变成node
    cur = build_node(num)
    #再加进入
    lists.append(cur)


def f(lists: list[Optional[Node]])-> Node:
    import heapq #优先级队列
    #优先级队列的用法是heapq.heappush( q,(比较的元组))
    q = []
    for i ,node in enumerate(lists):
        if node:
            heapq.heappush(q,(node.val,i,node))
    dummy = Node(val = 0)
    #从堆中弹出一个，next加入堆，
    ans = []
    while q:
        #弹一个
        val,i,node = heapq.heappop(q)
        ans.append(val)
        if node.next:
            heapq.heappush(q,(node.next.val,i,node.next))
    return ans 

print(" ".join(map(str,f(lists))))





