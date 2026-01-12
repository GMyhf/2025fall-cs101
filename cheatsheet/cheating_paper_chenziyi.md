大部分内容来自23工院 武昱达学长的cheatsheet
## 1. 语法
### 进制转化
```python
# 十进制转其他进制
x = 255
print(bin(x))   # 0b11111111
print(oct(x))   # 0o377
print(hex(x))   # 0xff

# 其他进制字符串转十进制
print(int('0b11111111', 2))   # 255
print(int('0o377', 8))        # 255
print(int('0xff', 16))        # 255
```
### 列表、集合、字典相关函数
#字典
```python
#字典 d
value = d.get('key', 'default')  # 安全访问 if key not in value: return 'default'
del d['key']  # 删除指定键
value = d.pop('key', 'default')  # 删除并返回值 if key not in value: return 'default'
for key in d: #遍历键
for key in d.values(): #遍历值
for key,value in d.items(): #遍历键值对
#集合 s
# 子集/超集判断
A = {1, 2, 3}
B = {1, 2, 3, 4, 5}
print(A <= B)  # True(若去掉=，则为真子集)
print(B >= A)  # True
print(A.isdisjoint(C))  # True（没有交集）
#列表lst
lst.insert(1, 1.5)  # 在索引1处插入1.5
lst.extend([5, 6])  # 批量添加（相当于 lst += [5, 6]）
del lst[0]  # 删除索引0的元素
sublist = lst[1:4:2]  # 带步长的切片
lst.sort(key=lambda x: x%3)  # 按自定义规则排序
# 统计和查找
count = lst.count(2)  # 统计元素出现次数：3
index = lst.index(3)  # 查找元素首次出现索引：2
index = lst.index(2, 2, 5)  # 在切片[2:5]中查找
#拷贝->不确定就用深拷贝
import copy
b = a  # 赋值，b和a是同一个对象
a[0] = 999
print(b)  # [999, 2, [3, 4]]，b也跟着变了
a = [1, 2, [3, 4]]
b = copy.copy(a)  # 浅拷贝->节省内存
# 修改外层：互不影响
a[0] = 999
print(a)  # [999, 2, [3, 4]]
print(b)  # [1, 2, [3, 4]]，b没变
# 修改内层：互相影响
a[2][0] = 999
print(a)  # [999, 2, [999, 4]]
print(b)  # [1, 2, [999, 4]]，b也跟着变了！
b = copy.deepcopy(a)  # 深拷贝
a[2][0] = 999
print(a)  # [1, 2, [999, 4]]
print(b)  # [1, 2, [3, 4]]，b不受影响

for index,value in enumerate([a,b,c]): # 每个循环体里把索引和值分别赋给index和value。如第一次循环中index=0,value="a"
key=next((key for key,value in d.items() if value==target),None) #根据值找到第一个对应的键
```

```python
q=float('(+/-)inf') #无穷，有正有负
```
### ASCII
```python
ord() # 字符转ASCII
chr() # ASCII转字符
```
### 保留n位小数
```python
pi = 3.1415926535
# 保留2位小数
print(f"{pi:.2f}")      # 3.14
# 保留5位小数
print(f"{pi:.5f}")      # 3.14159
# 保留0位小数（取整）
print(f"{pi:.0f}")      # 3
import math
x = 3.14159
# 向下取整（地板函数）
print(math.floor(x * 100) / 100)  # 3.14 (保留2位向下取整)
# 向上取整（天花板函数）
print(math.ceil(x * 100) / 100)   # 3.15 (保留2位向上取整)
```
## 2. 判断素数
### 筛法 - 判断较多数字是否为质数
```python
def Euler_sieve(n):
    primes = [True for _ in range(n+1)]
    p = 2
    while p*p <= n:
        if primes[p]:
            for i in range(p*p, n+1, p):
                primes[i] = False
        p += 1
    primes[0]=primes[1]=False
    return primes
#输出为True/False
```

```python
N=20
primes = []
is_prime = [True]*N
is_prime[0] = False;is_prime[1] = False
for i in range(2,N):
    if is_prime[i]:
        primes.append(i)
    for p in primes: #筛掉每个数的素数倍
        if p*i >= N:
            break
        is_prime[p*i] = False
        if i % p == 0: #这样能保证每个数都被它的最小素因数筛掉！
            break
#输出为小于N的素数
```

```python
# 埃氏筛 基本够用
N=20
primes = []
is_prime = [True]*N
is_prime[0] = False;is_prime[1] = False
for i in range(1,N):
    if is_prime[i]:
        primes.append(i)
        for k in range(2*i,N,i): #用素数去筛掉它的倍数
            is_prime[k] = False
print(primes)
```
### 判断较少数是否为质数
因为质数除了2，3都满足6k-1或6k+1
```python
#以6为步长
import math
def is_prime(n):
    if n <= 1: # 1 不是质数
        return False
    if n <= 3: # 2 和 3 是质数
        return True
    # 2 和 3 以外的偶数和能被 3 整除的数不是质数
    if n % 2 == 0 or n % 3 == 0:
        return False
    # 从 5 开始，步长为 6
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
```
## 3. 拓展包
#### math
```python
import math
print(math.pow(2,3)) # 8.0 乘方
print(math.pow(2,2.5)) # 5.656854249492381
print(math.sqrt(4)) # 2.0
print(math.log(100,10)) # 2.0  math.log(x,base) 以base为底，x的对数
print(math.comb(5,3)) # 组合数，C53
print(math.factorial(5)) # 5！
```
### bisect-只能在全部单调的列表中使用
```python
import bisect
sorted_list = [1,3,5,7,9] #[(0)1, (1)3, (2)5, (3)7, (4)9]
position = bisect.bisect_left(sorted_list, 6)
print(position)  # 输出：3，因为6应该插入到位置3，才能保持列表的升序顺序
bisect.insort_left(sorted_list, 6)
print(sorted_list)  # 输出：[1, 3, 5, 6, 7, 9]，6被插入到适当的位置以保持升序顺序
sorted_list=(1,3,5,7,7,7,9)
print(bisect.bisect_left(sorted_list,7))
print(bisect.bisect_right(sorted_list,7))
# 输出：3 6
```
### calendar & datetime
```python
print(calendar.isleap(2020)) #判断闰年
date3 = date(2023, 1, 1)
date4 = date(2023, 12, 31)
days_diff = abs((date4 - date3).days) #abs:获取绝对值
```
### counter
```python
from collections import Counter 
counter_result = Counter(data) # 返回一个字典类型的东西
# 输出统计结果
print(counter_result) # Counter({'apple': 3, 'banana': 2, 'orange': 1})
print(counter_result["apple"]) # 3
```
### heapq
堆擅长查找操作，但不擅长修改操作。一旦一个元素入堆，除非它运动到堆顶，不然我们以后就再也找不到它了。因此很多情况下，**堆要与其他数据结构配合使用**，而其他数据结构的主要任务就是标记。-->dijkstra
```python
>>> import heapq
>>> heap=[8,7,6,5,4,3,2,1]
>>> # 初始化一个堆（heapq.heapify-->将列表*原地*转化为堆）
>>> heapq.heapify(heap)
>>> >>> # 严正警告：heapq.heapify操作返回None，绝对不要把它赋值给堆！
>>> heap=heapq.heapify([8,7,6,5,4,3,2,1])
>>> print(heap)
None

>>> heap
[1, 4, 2, 5, 8, 3, 6, 7]
>>> # 将元素加入堆中
>>> heapq.heappush(heap,-1)
>>> heap
[-1, 1, 2, 4, 8, 3, 6, 7, 5]
>>> # 弹出堆顶元素
>>> heapq.heappop(heap)
-1
>>> heap
[1, 4, 2, 5, 8, 3, 6, 7]
```
### 判断完全平方数
```python
import math
def isPerfectSquare(num):
    if num < 0:
        return False
    sqrt_num = math.isqrt(num)
    return sqrt_num * sqrt_num == num
print(isPerfectSquare(97)) # False
```
## 4. 算法
### ==dp==
#### 1.背包问题
1. **01背包** --> 每个物品只能拿一次
```python
#小偷背包
def zero_one_bag():
    # V-总容量,n-物品个数,
    # 各物品体积cost=[0,     ],价格price=[0,     ]
    dp=[0]*(V+1) 
#初始化：如果要求恰好装满则为[0,-inf,……]；如果可以装不满为[0]*(V+1) <-- 如果要求恰好装满，则除了容量为0的背包在什么都不装的条件下就满足题意，其他容量的背包只有恰好能装满才能认为是有效的（正数）；而若不要求恰好装满，则任何容量的背包都可以合法存在，只要满足装的价值最大。
    for i in range(1,n+1):           #每个物品
        for j in range(V,cost[i]-1,-1):    #逆向遍历每个容量-->保证在更新dp[j]时，dp[j-cost[i]]还没有被更新过
            dp[j]=max(dp[j],price[i]+dp[j-cost[i]])
    return dp[-1]
```
2. **完全背包** --> 每个物品可以拿无限次
```python
#零钱找零
def total_bag():
    dp=[0]*(V+1)
    for i in range(1,n+1):           #每个物品
        for j in range(1,cost[i]+1):       #正向遍历每个容量
            dp[j]=max(dp[j],price[i]+dp[j-cost[i]])
    return dp[-1]
```
3. **多重背包** --> 每个物品的个数有限制,把每个物品的个数拆成1 2 4等转化为01背包
```python
#NBA门票
def many_bag():
    #s=[0,   ]为每个物品的个数
    dp=[0]*(V+1)
    for i in range(1,n+1):
        k=1 #将s[i]用2的幂次和进行表示，进而将s[i]个物品组装成多个物品组，且各物品组的互相结合可以组装出s[i]个物品能组装出的所有可能性
        while s[i]>0:
            cnt=min(k,s[i]) #进行组合
            for j in range(V,cnt*cost[i]-1,-1):
                dp[j]=max(dp[j],cnt*price[i]+dp[j-cnt*cost[i]])
            s[i]-=cnt
            k*=2
    return dp[-1]
```
4. **二维费用背包**（每个物品都有两种不同的费用（比如重量和体积），背包对于这两种费用都有最大限制（比如最大承重和最大容积）。我们需要选择一些物品放入背包，使得在满足两种限制的条件下，获得的总价值最大。）
```python
def two_dimension_cost(n,V1,V2,cost1,cost2,price):
    dp=[[0]*(V2+1) for _ in range(V1+1)]
    for i in range(n):
        for c1 in range(V1-1,cost1[i]-1,-1): #对两个维度分别遍历
            for c2 in range(V2-1,cost2[i]-1,-1):
                dp[c1][c2]=max(dp[c1][c2],dp[c1-cost1[i]][c2-cost2[i]]+price[i])
    return dp[-1][-1]
```
---
#### 2.整数分割问题
1. 把n划分为若干个正整数，**不考虑顺序** --> 完全背包
4：4=3+1=2+2=2+1+1=1+1+1+1 共5种
```python
def divide1(n):
    dp=[1]+[0]*n    #把0划分只有0这一种
    for i in range(1,n+1):           #每个数字
        for j in range(i,n+1): #正向遍历每个容量（每个n）；不考虑顺序-->从i（包括i）开始遍历，i之前的已经遍历过了
            dp[j]+=dp[j-i]
    return dp[-1]
```
2. 把n划分为若干个正整数，**考虑顺序**
4：4=3+1=1+3=2+2=2+1+1=1+2+1=1+1+2=1+1+1+1 共8种
```python
def divide2(n):
    dp=[1]+[0]*n
    for i in range(1,n+1):           #每个容量（每个n）
        for j in range(1,i+1):             #每个可能划分出的数字
            dp[i]+=dp[i-j]
    return dp[-1]
```
3. 把n划分为若干个不同的正整数，不考虑顺序 --> 01背包
4：4=3+1 共1种
```python
def divide3(n):
    dp=[1]+[0]*n
    for i in range(1,n+1):
        for j in range(n,i-1,-1):
            dp[j]+=dp[j-i]
    return dp[-1]
```
4. 把n划分为k个正整数，不考虑顺序
```python
#放苹果
#dp[n][k]:把n分成k组
def divide4(n,k):
    dp=[[0]*(k+1) for _ in range(n+1)] 
    #每个数字分成1组都是1种
    for i in range(n+1):
        dp[i][1]=1
    for i in range(1,n+1):
        for j in range(2,k+1):
            #i<j时无法划分
            #i>=j时分为两种：若分组中有1，则为dp[i-1][j-1]
            #若无1，先把每组放进去1，则为dp[i-j][j]
            if i>=j:
                dp[i][j]=dp[i-1][j-1]+dp[i-j][j]
    return dp[n][k] #dp[-1][-1]
```
### ==并查集==
```python
def origin(n):  #对树（林）中的n个节点的父节点进行初始化
    return [i for i in range(n)]  
def find_father(a,father):  
    if father[a]==a:  #父节点是其本身--> 该结构的最终代表节点
        return a  
    else:  
        p=find_father(father[a],father)  #递归调用父亲的父亲
        father[a]=p  #压缩，减小时间复杂度
        return p  
def union(a,b,father):  
    f_a=find_father(a,father)  
    f_b=find_father(b,father)  
    if f_a!=f_b:  
        father[f_a]=f_b  
  
def main():  
    n,m = map(int,input().split())  
    father=origin(n)  
    for _ in range(m):  
        a,b = map(int,input().split())  
        union(a-1,b-1,father)  
    k=int(input().strip())  
    for _ in range(k):  
        a,b = map(int,input().split()) #查找a、b是否在同一个集合里 
        if find_father(a-1,father) != find_father(b-1,father):  
            print('No')  
        else:  
            print('Yes')  
if __name__ == '__main__':  
    main()
```
### ==Dilworth Theory==
**最少单调链个数 == 最长反单调链长度**
找最长上升子序列的长度，用left
找最长下降子序列，先reverse，再用left
如果是不降，用right
如果是不升，先reverse，再用right
看题目要求的最终结果是否需要相同元素的考虑，需要考虑用left，不需要用right
```python
from bisect import bisect_left,bisect_right
def d(s): #求最长上升子链长度
    lst=[]
    for i in s:
        pos=bisect_left(lst,i)
        if pos<len(lst):
            lst[pos]=i #为何可以直接替换？-->在i之前的元素构成的子序列已经有当前lst的长度，因此替换不会造成最终结果更长。但可以最小化该位置的值，为后续填数字创造更多可能性
        else:
            lst.append(i)
    return len(lst)
```
### ==PREFIX SUM==
#### 1.前缀和数组
用于处理**多次查询**从[l,r]的序列之和的问题
```python
s=[int(i) for i in input().split()]
prefix=[s[0]]+[0]*(len(s)-1)
for i in range(1,len(s)):
    prefix[i]=prefix[i-1]+s[i]
distance_l_r=prefix[r]-(prefix[l-1] if l-1>=0 else 0)
```
#### 2.前缀和的特殊用法(哈希表)
使用prefix和prefix_map来记录已有的前缀和，从而判断子串和为0的子串个数；或找相同前缀和数字出现的最远位置
```python
#找出**不重叠**的和为0的子序列个数，一旦找到就将prefixed集合清空，
#cf-Kousuke's Assignment-2033D
t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
 
    prefix = 0
    prefixed = {0}
    cnt = 0
    for i in a:
        prefix += i
        if prefix not in prefixed:
            prefixed.add(prefix)
        else:
            cnt += 1
            prefix = 0
            prefixed={0}
    print(cnt)
```
### ==SORTING==
#### 1.冒泡排序
```python
def bubble_sort(s):
    n=len(s)
    f=True
    for i in range(n-1): #循环，直到1~n的所有数都成功冒泡
        f=False
        for j in range(n-i-1): #将n-i-1前的最大数冒泡到末尾
            if s[j]>s[j+1]:
                s[j],s[j+1]=s[j+1],s[j]
                f=True
        if f==False:
            break
    return s
```
#### 2.归并排序 --> 递归
```python
#每一层：将两半分别排序，利用merge(合并有序数组)函数组装到一起
#对每一层来说，再把本层对半取，进行相同操作
def merge_sort(s):
    if len(s)<=1:
        return s
    mid=len(s)//2
    left=merge_sort(s[:mid])
    right=merge_sort(s[mid:])   #两次递归放在一起，与 hanoi tower 的递归以及 lc-LCR085-括号生成 的递归很相似
    return merge(left,right)
def merge(l,r):
    ans=[]
    i=j=0
    while i<len(l) and j<len(r):
        if l[i]<r[j]:
            ans.append(l[i])
            i+=1
        else:
            ans.append(r[j])
            j+=1
    ans.extend(l[i:])
    ans.extend(r[j:])
    return ans
```
#### 3.快速排序 --> 递归，选基准
```python
def quick_sort(s):
    if len(s)<=1:
        return s
    base=s[0]
    left=[x for x in s[1:] if x<base]
    right=[x for x in s[1:] if x>=base]
    return quick_sort(left)+[base]+quick_sort(right)
```
### ==searching==
#### 1. dfs
dfs如果要解决枚举类的题目通常会涉及回溯操作，而在原地修改时可能无需回溯。如果有回溯操作必须要有退出条件。
如果dfs内部有类似于dp数组需要不断访问某些元素的值的时候，除了开空间创建一个dp，还可以用lru_cache。
但一定要在需要进行记忆化递归的函数头顶上写，否则无效。
```python
from functools import lru_cache
@lru_cache(maxsize=2048) #或者更大，如None，考虑内存因素自行调整
def dfs():
    ...
```
1. 无回溯 - 原地修改
2. 有回溯 - 要创建拷贝，不能直接引用！！
```python
#oj-八皇后-02754
'''考虑以下递归步骤：
在某次递归时，curr = [1, 5, 8, 6]，此时 ans.append(curr)。
接下来，回溯修改了 curr，变为 [1, 5, 8, 7]。
由于 ans 中保存的是 curr 的引用，ans 中原本存储的 [1, 5, 8, 6] 也会变为 [1, 5, 8, 7]。
因此使用 curr[:]，创建当前列表的拷贝，确保后续对 curr 的修改不会影响已保存的解
'''
visited=[0]*8
ans=[]
def dfs(k,curr): #k - 递归深度
    global ans #声明ans是全局变量，以使递归深度达到时，真正在全局范围内修改ans
    if k==9:
        ans.append(curr[:]) #curr[:]VScurr --> 创建拷贝
        return
    for i in range(1,9):
        if visited[i-1]:
            continue
        if any(abs(j-i)==abs(len(curr)-curr.index(j)) for j in curr):
            continue
        visited[i-1]=1
        curr.append(i)
        dfs(k+1,curr)
        visited[i-1]=0
        curr.pop() #回溯
dfs(1,[])
# print(ans)
for _ in range(int(input())):
    n=int(input())
    print(''.join(map(str,ans[n-1])))
```
#### 2. bfs
模板：
```python
from collections import deque  
def bfs(start, end):    
    q = deque([(0, start)])  # (step, start)
    in_queue = {start}
	direction=[(dx,dy)……]
    while q:
        step, front = q.popleft() # 取出队首元素
        if front == end:
            return step # 返回需要的结果，如：步长、路径等信息
		for dx,dy in direction:
			nx=dx+front[0]
			ny=dy+front[1]
			if nx ny 均满足有效范围 and (nx,ny) not in in_queue:
				in_queue.add((nx,ny))
				q.append((step+1,(nx,ny)))				
        # 将 front 的下一层结点中未曾入队的结点全部入队q，并加入集合in_queue设置为已入队 
```
#### 3. Dijkstra - 处理带权图（仅在没有负权边时使用）
```python
#oj-走山路-20106
import heapq
dx,dy=[0,-1,1,0],[-1,0,0,1]
def dijkstra(sx,sy,ex,ey):
    if s[sx][sy]=='#' or s[ex][ey]=='#':
        return 'NO'
    q=[]
    dist=[[float('inf')]*m for _ in range(n)]
    heapq.heappush(q,(0,sx,sy)) #(distance,x,y)
    dist[sx][sy]=0
    while q:
        curr,x,y=heapq.heappop(q) #heappop()-->弹出元组[0]最小的组，即到达起点最近的点
                if (x,y)==(ex,ey):
                return curr
        for i in range(4):
            nx,ny=x+dx[i],y+dy[i]
            if 0<=nx<n and 0<=ny<m and s[nx][ny]!='#':
                new=curr+abs(s[x][y]-s[nx][ny])
                if new<dist[nx][ny]:
                    heapq.heappush(q,(new,nx,ny)) #heappush()-->若新的起点到达该点的路程小于原有的起点到达该点的路程，则在q中更新替代，在dist中同样更新替代
                    dist[nx][ny]=new
    return 'NO' #遍历了一遍仍未到达终点-->到不了终点
```
### ==二分查找==
在进行二分之前一般需要对列表进行排列。在一类特殊题中与greedy结合，如表述为**求最大值中的最小值**。
例题：
```python
#oj-aggressive cows-02456
def binary_search():
    l=0
    r=(s[-1]-s[0])//(c-1)
    while l<=r: #<=
        mid=(l+r)//2
        if can_reach(mid):
            l=mid+1
        else:
            r=mid-1
    return r #r
def can_reach(mid):
    cnt=1
    curr=s[0]
    for i in range(1,n):
        if s[i]-curr>=mid:
            cnt+=1
            curr=s[i]
    return cnt>=c
```
## 5. RE的108种死法
1. 除以零（ZeroDivisionError） 
2. 列表索引越界（IndexError）
3. 使用不存在的字典键（KeyError）
4. 递归深度超过限制（RecursionError）
5. 内存超限（MemoryError，可能由于创建了太大的列表等）
6. 使用未赋值的变量（NameError）
7. 类型错误（TypeError）
8. 值错误（ValueError）
9. 栈溢出（递归太深）