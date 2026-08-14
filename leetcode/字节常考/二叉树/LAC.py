

class TreeNode:
    def __init__(self,val = 0,left = None,right = None):
        self.val = val 
        self.left = left 
        self.right = right 

from collections import deque 
def buildTree(t):
    n = len(t)
    q = deque()
    head = TreeNode(val = int(t[0]))
    q.append(head)#先把第一个加进去
    if n == 1:
        return head 
    i = 1
    
    #当前层的节点
    while q:
        node = q.popleft()
        if i <n:
            if t[i] != "null":#处理左边的
                node.left = TreeNode(val = int(t[i]))
                q.append(node.left)
            i+=1
        if i < n:
            if t[i] !="null":
                node.right = TreeNode(val = int(t[i]))
                q.append(node.right)
            i+=1
    return head


def LCA(root,p,q):
    if not root or root == p or root == q:
        return root
    left = LCA(root.left, p, q)
    right = LCA(root.right, p, q)

    if left and right:
        return root 
    if not left and not right:
        return None 
    return left if left else right 

#找到p和q，然后弄成tree node
def find_node(root,val):
    if not root:
        return None 
    if root.val == val:
        return root 
    left = find_node(root.left,val)
    if left:
        return left 
    return find_node(root.right,val )

t = input().strip().split()
p_val,q_val = map(int,input().split())
root = buildTree(t)
p,q = find_node(root,p_val),find_node(root,q_val)
print(LCA(root,p,q).val)


