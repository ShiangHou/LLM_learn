# 二叉搜索树中第k小的元素
给定一个二叉搜索树的根节点 root ，和一个整数 k ，请你设计一个算法查找其中第 k 小的元素（k 从 1 开始计数）。


## 初步思路
二叉搜索树的特性依旧是中序遍历是递增的，直接中序遍历就好，第k小的就是取第k个的就行
```python 
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        if not root:
            return None
        # 先中序遍历
        root_list = []
        def f(node):
            nonlocal root_list
            if not node:
                return 
            f(node.left)
            root_list.append(node.val)
            f(node.right)
        f(root)
        return root_list[k-1]

```