n = int(input())
nums = list(map(int,input().split())) if n!= 0 else []
k = int(input())

class Node:
    def __init__(self,val = 0,next = None):
        self.val = val
        self.next = next 


def build_node(nums,n):
    tail = Node(val = nums[-1])
    cur = tail
    for i in range(n-2,-1,-1):
        temp = Node(val = nums[i],next = cur)
        cur = temp
    return cur 

def f(node,n,k ):
    if k == n:
        return node.next
    
    
    head = node
    slow,fast = node,node
    while k >0:
        fast = fast.next
        k-=1
    while fast.next:
        fast = fast.next
        slow = slow.next
    
    slow.next = slow.next.next 
    return head 

node = build_node(nums,n)
ans_node = f(node,n,k )


result = []
while ans_node:
    result.append(ans_node.val)
    ans_node = ans_node.next
print(" ".join(map(str,result)))
