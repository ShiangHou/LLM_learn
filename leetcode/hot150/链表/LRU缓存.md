# LRU缓存
请你设计并实现一个满足  LRU (最近最少使用) 缓存 约束的数据结构。
实现 LRUCache 类：
LRUCache(int capacity) 以 正整数 作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字 key 已经存在，则变更其数据值 value ；如果不存在，则向缓存中插入该组 key-value 。如果插入操作导致关键字数量超过 capacity ，则应该 逐出 最久未使用的关键字。
函数 get 和 put 必须以 O(1) 的平均时间复杂度运行。

 

示例：

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

## 初步思路

首先我们需要准备的数据结构是 双端链表，以及一个hash，hash的key是链表的val，value是node

此外我们需要准备两个辅助函数，一个是叫做removenode，就是把这个节点删除，一个是addtohead，就是把一个节点移动到最前面

组合技就是removetohead，把一个用过的node删除后移动到最前面

get方法就是直接按照value从里面取，取完后这个removetohead就行

put方法就是先查，然后有的话就返回，removetohead，没有的话就加在头部，然后看一下尺寸，

```python 
class Node:
    def __init__(self,key = 0,val = 0,prev = None,next = None):
        self.key = key#题目要求找的是关键词以及对应的val。所以这样弄
        self.val = val
        self.prev = prev
        self.next = next



#正式
class LRUCache:

    def __init__(self, capacity: int):
        #初始化
        self.cache = {}#初始化hash

        #初始化虚拟头尾
        self.head = Node()
        self.tail = Node()

        #收尾相连
        self.head.next = self.tail
        self.tail.prev = self.head

        self.size = 0
        self.capacity = capacity
    #写两个辅助函数
    def removenode(self,node):#益处node
        #很简单，node的前一个指向node的下一个，node的后一个指向前一个
        node.prev.next = node.next
        node.next.prev = node.prev
    def addtohead(self,node):#把一个node加到head前面
        #很简单，先把这个node接上，然后处理老的
        #先接node
        node.prev = self.head
        node.next = self.head.next

        #处理老的
        self.head.next.prev = node#原来head后面的 接上node
        self.head.next = node#head后面接上node
    #组合机
    def removetohead(self,node):
        self.removenode(node)
        self.addtohead(node)


    def get(self, key: int) -> int:
        if key in self.cache:
            #如果在了
            node =  self.cache[key]#获取node
            self.removetohead(node)#移走放在第一个
            return node.val
        #不在，直接返回-1
        else:
            return -1
        

    def put(self, key: int, value: int) -> None:
        if key in self.cache:#如果在了，直接改就行
            node = self.cache[key]
            node.val = value#改值
            self.removetohead(node)#移动
        else:#不在
            node = Node(key = key,val = value)
            #加到cache和head
            self.cache[key] = node
            self.addtohead(node)
            self.size +=1
            if self.size > self.capacity:
                #获取队尾元素，益处
                tail_node = self.tail.prev
                self.removenode(tail_node)
                #移除cache的
                del self.cache[tail_node.key]
                self.size -=1
            


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)

```



