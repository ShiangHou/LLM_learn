nums = list(map(int,input().split()))

# def f(nums):
#     ans = 1
#     n = len(nums)
#     dp = [1]*n 
#     for i in range(n):
#         for j in range(i):
#             if nums[i] > nums[j]:
#                 dp[i] = max(dp[i],dp[j]+1)
#     return max(dp)

def bisearch(tail,x):
    #找到比最左边第一个比x大的，返回下标
    n = len(tail)
    l,r = 0,n-1
    while l < r:
        mid = (l+r) // 2
        if tail[mid] >= x:
            r = mid
        else:
            l = mid+1 
    return r 


def f(nums):
    tail = []
    n = len(nums)
    for i in range(n):
        if not tail or nums[i] < tail[-1]:
            tail.append(nums[i])
        else:
            index = bisearch(tail,nums[i])
            tail[index]  = nums[i]
    

    return len(tail)




print(f(nums))