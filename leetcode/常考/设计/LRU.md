# LRU 缓存

题目描述

首先题目说，我需要实现一种数据结构

初始化 这个缓存可以装多少东西

get操作，入参是key，就是获取一个key，如果这个key在里面，那么就返回对应的value（所以说是一个key value的字典）

put 操作，入参是key value，意思就是说我把key以及对应的value放入，如果key有的话，就把它的value变成我现在要插入的value，如果没有的话，就插入一个key-value，如果满的话，就应该逐出最久没有使用的

函数get和put必须是O1的复杂度

## 双端链表

首先既然是key和value，而且又要这个O1的复杂度，那么肯定是用一个哈希字典来去弄，

那为什么要用双端链表呢

因为hash是无序的，我们不知道哪一个key是刚刚访问过的，而哪一个key是在角落里里面的

所以说，我们需要把所有的key和value都排队，当每一次使用的时候，我们就把用的放到队伍前面就行，如果要挤出，就直接让队尾的走就行

并且，为了让这个过程是O1的，我们不能用数组（因为会逐个），单向链表也不太行，所以我们选择双向链表

那么我们的hash肯定也要进行改变，怎么改变呢，我们直接把value变成一个node！！！

这样，我们的key就指向了那个node，然后node和node之前就用双向队列排排号就行

如果我要get的话，我有key，那么通过key可以直接连接到那个node的value

然后由于有node，双向链表，我们直接把node抠出来，放到链表的头节点就行，因为我们没有for循环，所以时间还是O1的

python中实现一个双端链表就是

```python 
class Node:
    def __init__(self,key = 0,value = 0,prev = None,next = None):
        self.key = key
        self.value = value
        self.prev = prev
        self.next = next
```

## 流程

首先，我们需要两个虚拟的节点，一个是头，一个是尾，没啥作用，主要是标记边界，

初始化的时候，让head指向tail就行

### 实现双端链表

python中实现一个双端链表就是

```python 
class Node:
    def __init__(self,key = 0,value = 0,prev = None,next = None):
        self.key = key
        self.value = value
        self.prev = prev
        self.next = next
```

### 初始化

```python 
class LRU:
    def __init__(self,capacity):
        self.cache = {}#就是存放hash的

        #存放虚拟的头和虚拟的尾
        self.head = None
        self.tail = None


        #让然后让头指向尾，尾指向头
        self.head.next = self.tail
        self.tail.prev = self.head

        #以及总长size
        self.size = 0

```


### 辅助函数

我们需要写两个辅助函数，

remove：把节点 node 从双向链表中“抠”出来，让它前后的节点直接握手

很简单，node的前一个要和node的后一个连着，即 node.prev.next = node.next

然后node的后一个要和node的前一个连着，即 node.next.prev = node.prev

```python 
def removenode(self,node):
    node.prev.next = node.next
    node.next.prev = node.prev
```

另一个函数是 addtohead，就是把一个新的node放在head的后面

思想是，先把新的左右弄好，然后弄之前的
```python 
def addtohead(self,node):
    node.prev = self.head#先安顿好新人
    node.next = self.head.next

    #在弄老人
    self.head.next.prev = node#原来head后面那个节点的前面指向node
    self.head.next = node
```

我们其实可以弄一个组合技，就是把已存在的节点放在头

```python 
def movetohead(self,node):
    self.removenode(node)
    self.addtohead(node)


```

### 最终

最后，我们的get和put其实就是

get：从self.cache里面拿到node，在旧位置拔出来，放到新位置去（也就是remove和addtohead），最后return一下node.value

然后put是，如果是新key，就newnode = node（key,value）,然后addtohead

如果满了，直接找到最后一个，让它remove就行，即

lastnode = tail.prev，remove lastnode

#### get
查表  没查到报错   查到了就把它移到最前面，并返回它的值。



我们其实可以弄一个组合技，就是把已存在的节点放在头

```python 
def removetohead(self,node):
    self.removenode(node)
    self.addtohead(node)

def get(self,key):
    if key not in self.cache:
        return -1
    node = self.cache[key]

    self.removetohead(node)#因为刚被访问过，所以删除，然后加到头

    return node.val

```


#### put

```python 
def removetail(self,node):
    #淘汰最后的
    node = self.tail.prev
    self.remove(node)

    return node

def put(self,key,value):
    #情况一：已经存在了
    if key in self.cache:
        node = self.cache[key]
        node.val = value

        #操作过，移到头部
        addtohead(node)
    #情况二，不存在
    else:
        node = Node(key,value)#创立新的节点
        addtohead(node)#放到头部
        self.cache[key] = node#放入hash
        self.size += 1
        #看一下有没有超过去
        if self.size > self.capacity:
            #移除最后一个
            removedNode = self.removeTail()

            #把hash也移除
            del self.cache[removeNode.key]

            self.size -= 1




```