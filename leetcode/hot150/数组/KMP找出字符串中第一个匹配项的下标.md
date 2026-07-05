# 找出字符串中第一个匹配项的下标
给你两个字符串 haystack 和 needle ，请你在 haystack 字符串中找出 needle 字符串的第一个匹配项的下标（下标从 0 开始）。如果 needle 不是 haystack 的一部分，则返回  -1 

## 初步思路

标准的KMP算法

首先先看主程序

i是目标的（就是长的），j是短的

i和j都从0开始，
如果j是-1（就是取next的时候取到了next的第一个）或者是i和j位置的相等，此时i++，j++

否则，遇到不想等了，直接让j移动到nextj的位置

最后，如果j是到了最后的（即j==m的长度），返回i-j就行，如果没到，那么就说明没匹配，返回-1


next数组怎么设立

next数组的含义是 最长的公共前后缀

next数组的实现也是very的巧妙，有点像是动态规划

首先对于一个位置？，先看它上一个的next数组在哪，比如在7位置，如果7位置的值==？位置的值，那么说明相等，？位置的next就是7+1

如果不等的话，直接跳，比如7位置的！= ？位置的，用7位置的next值，比如是3，看3位置和7位置的相等嘛，相等的话？的next就是3+1为4

如果没挑出来，就是0

代码的思路是，首先先算0位置和1位置的，0位置的就是-1，1位置的就是0，i从2位置开始

准备一个变量cn，

cn = 0#cn表示当前要和前一个字符对比的下标（就是i-1位置的next的指向的，由于i从2开始，1位置的next是0，所以这里也是0）

先看i-1和cn位置是不是一样的，是一样的话直接cn+1，然后ans[i] = cn(相当于之前说的那个位置+1),然后i+1

如果不是，cn>0,继续跳，即cn = ans[cn]

如果cn<0,那么跳不动了，直接ans = 0就行，然后i滑动

```python 
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:


        def get_next(s):
            if len(s) == 1:
                return [-1]
            ans = [0]*len(s)
            ans[0] = -1
            i = 2
            cn = 0
            while i <len(s):
                if s[i-1] == s[cn]:
                    cn+=1
                    ans[i] = cn
                    i+=1
                elif cn>0:
                    #跳
                    cn = ans[cn]
                else:
                    ans[i] = 0
                    i+=1
            return ans

        #主流程
        n = len(haystack)
        m = len(needle)
        i,j = 0,0
        _next = get_next(needle)
        while i <n and j<m:
            if j == -1 or haystack[i] == needle[j]:
                i+=1
                j+=1
            else:
                #直接让j移动到nextj的位置
                j = _next[j]
        return i-j if j == m else -1


        

```