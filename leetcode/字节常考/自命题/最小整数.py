'''
题目描述
‍‌​​​‌​​​‌​​​‌​​‌​‌‍ 给定一个数组nums，和一个整数s(不存在前导0），请你从

nums中选择一些整数（可以重复使用），使得组成的数字是小于

s的整数，且要求尽可能的大。

输出最终构成的数字，如果不存在满足条件的整数，则输出-1。
输入描述
第一行输入一个整数 
n(1≤n≤10) ，表示数组
nums的元素个数

第二行输入

n个整数

nums[1],nums[2],...nums[n](0≤nums[i]≤9)

第三个输入一个整数

s(1≤s≤10 
10000
 )

输出描述
输出符合条件的最小整数（不能含有前导0，例如00256应该输出256），如果不存在满足条件的整数，输出-1。

样例1
输入

4
5 4 8 2
5416
复制
输出

5288
复制
样例2
输入

3
1 2 3
111
复制
输出

33

'''
'''
思路就是
从左往右：

能相等 → 相等，继续
不能相等但能变小 → 选最大的较小值，后面全最大
连较小值都没有 → 向前回退

如果最后完全等于 s → 也要回退

回退不了 → 位数减 1，直接构造最大数字


'''

n = int(input())
nums = list(map(int,input().split()))
s = int(input())

digit = sorted(set(nums),reverse= True)

def main(digit,s):
    max_digit = digit[0]#最大值，方便后续直接填充

    #写两个辅助函数，一个是找小于等于的，一个是找小于的，没有的话返回None，然后入参加一个first，判断是不是第一个，然后过滤掉0
    def leq(x,first = False):#小于等于
        for q in digit:
            if first and q == 0:#遇到第一个是0的跳过
                continue
            if q <= x:
                return q 
        return None


    def less(x,first = False):#严格小于
        for q in digit:
            if first and q == 0:#遇到第一个是0的跳过
                continue
            if q < x:
                return q 
        return None
    #正式开始
    #准备一个东西存数
    ans = []
    for i in range(n):
        x = int(s[i])
        d = leq(x,i == 0)
        if d == None:#没有比这个位置大于等于的
            break#跳出去，待会回退
        ans.append(d)
        if d <x :#如果这个东西是比x小的，不是贴着等于，那么后面所有直接用最大值填上就行
            ans += [max_digit] *(n-i-1)#后面还剩n-i-1个
            return ans 
        

    #回退，既可以承接break，也可以承接答案一样的时候
    j = len(ans) -1
    while j >= 0:
        #找到小的
        d = less(ans[j],j == 0)
        if d is not None:#找到了一个小的，直接把它换了，再把从他之后全部的弄成最大的
            ans[j] = d 
            ans = ans[:j+1]
            ans += [max_digit]*(n-j-1)
            return "".join(map(str,ans))
        j -=1
    
    #如果上面的while没跑完，说明没有和这个长度相同的答案，所以应该找到最大非0数的第一位，换掉，然后后面跟最大的就行
    first_digit = None
    for q in digit:
        if q != 0:
            first_digit = q
            break 
    if n > 1 and first_digit is not None:
        #直接拼答案
        ans = [first_digit] + max_digit*(n-2)
        return "".join(map(str,ans))
    if 0 in digit:
        return "0"
    return -1

print(main(digit,s))





    

    



    

