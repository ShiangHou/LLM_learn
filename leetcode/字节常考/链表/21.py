t = int(input())
for _ in range(t):
    n1 = int(input())
    l1 = list(map(int,input().split()))
    n2 = int(input())
    l2 = list(map(int,input().split()))


class Node:
    def __init__(self,val = 0,next = None):
        self.val = val
        self.next = next

def build_node(nums,n):
    if n == 0: return None
    tail= Node(nums[-1],None)
    cur = tail
    for i in range(n-2,-1,-1):
        temp = Node(val = nums[i],next = cur)
        cur = temp
    return cur 
