# Z字型变换
将一个给定字符串 s 根据给定的行数 numRows ，以从上往下、从左到右进行 Z 字形排列。

比如输入字符串为 "PAYPALISHIRING" 行数为 3 时，排列如下：

P   A   H   N

A P L S I I G

Y   I   R
之后，你的输出需要从左往右逐行读取，产生出一个新的字符串，比如："PAHNAPLSIIGYIR"。

请你实现这个将字符串进行指定行数变换的函数：

string convert(string s, int numRows);
 

示例 1：

输入：s = "PAYPALISHIRING", numRows = 3
输出："PAHNAPLSIIGYIR"

## 初步思路

额，看了一会题目才看懂

这里说的是，把一个字符串用这种方式去转，然后的话读取的时候用转完的去读取

感觉是个数学题？？

好像不是

看一下题解，very的巧妙

首先是，我们最后的z有三个是吧，就是第一行，第二行和第三行，假设取名叫做res[0],res[1],res[2]

最终，我们需要遍历一遍原始的数据，一行一行的填上去

比如说leetcode

l填res[0]的数组，e填res[1]的数组，e填res[2]的数组，此时弄一个flag，把t填到res[1]

然后反过来c填res[0],o填res[1],e填res[2]，这样就可以

最后拼起来这三个就可以

在这个过程中，我们需要维护一个flag，因为numRows不是固定的

比如4个

输入：s = "PAYPALISHIRING", numRows = 4
输出："PINALSIGYAHRPI"
解释：
P     I    N
A   L S  I G
Y A   H R
P     I

所以流程其实是，首先，我们需要维护两个变量，一个是i，一个是flag，
i会从0到numRows然后再从numRows到0

i通过每次加flag来去控制

每次填的时候，i都会+flag

flag就是1或者-1，等i到num的数量时flag转换成-1

切换到-1的时候，此时就是往后填写就行

```python 
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        if numRows: < 2:
            return s
        #先建立numRow的数组,这里考虑用一个二维的,一共num个空字符串
        res = ["" for _ in range(numRows)]
        i,flag = 0,-1
        for c in s:
            #先加一个，不管头和尾巴，
            res[i] += c
            if i == 0 or i == numRows-1:
                flag = -flag#所以最初flag要从-1开始，相当于之前的流程已经转到头了
            i+=flag
        return "".join(res)





```