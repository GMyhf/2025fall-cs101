## 1.常用技巧
```python
from functools import lru_cache 
@lru_cache(maxsize = 128) +函数

import sys
sys.setrecursionlimit(1 << 30)

import sys
input == sys.stdin.readline
#加快读取速度

import math  
print(dir(math))  
print(help(math))  
print(help(math.comb))
#查看模块及函数说明

print(bin(9)) #bin函数返回二进制，形式为0b1001
print(oct(9)) #八进制
print(hex(9)) #十六进制
print(int(binary_str, 2)) # 第一个参数是字符串类型的某进制数，第二个参数是他的进制，最终转化为整数

dict.items()#同时调用key和value
dict.get(key,default=None) # 其中，my_dict是要操作的字典，key是要查找的键，default是可选参数，表示当指定的键不存在时要返回的默认值

print(round(3.123456789,5)# 3.12346
print("{:.2f}".format(3.146)) # 3.15

ord() # 字符转ASCII
chr() # ASCII转字符
a="hello world"
print(a.upper())#全部转为大写字母
print(a.lower())#全部转为小写字母
print(a.capitalize())#第一个字母转换为大写
print(a.title())#每个字母首字母转换为大写
print(a.swapcase())#全部转为小写字母
if(a.isupper())#是否大写
if(a.is_lower())#是否小写

for index,value in enumerate([a,b,c]): # 每个循环体里把索引和值分别赋给index和value。如第一次循环中index=0,value="a" 

# pylint: skip-file

while(True):
    try:

    except EOFError:
        break
        
# 位运算
& 与 两个位都为1时，结果才为1
| 或 两个位都为0时，结果才为0
^ 异或 两个位相同为0，相异为1
~ 取反 0变1，1变0
<< 左移 各二进位全部左移若干位，高位丢弃，低位补0
>> 右移 各二进位全部右移若干位，高位补0或符号位补齐
降低常数：
<<1 == *2
>>1 == //2
# 位运算判断是否为二的幂
t=int(input())
for i in range(t):
    a=int(input())
    print("YES" if a&(a-1) else "NO")

import statistics
median = statistics.median(data) #求中位数

from copy import deepcopy
a=deepcopy(b) #二维数组深拷贝
a=b.copy()#一维数组深拷贝
```
## 2. 拓展包
### (1)math
```python
import math
print(math.ceil(1.5)) # 2
print(math.pow(2,3)) # 8.0
print(math.pow(2,2.5)) # 5.656854249492381
print(9999999>math.inf) # False
print(math.sqrt(4)) # 2.0
print(math.log(100,10)) # 2.0  math.log(x,base) 以base为底，x的对数
print(math.comb(5,3)) # 组合数，C53
print(math.factorial(5)) # 5！
```
### (2)lru_cache
```python
# 需要注意的是，使用@lru_cache装饰器时，应注意以下几点：
# 1.被缓存的函数的参数必须是可哈希的，这意味着参数中不能包含可变数据类型，如列表或字典。
# 2.缓存的大小会影响性能，需要根据实际情况来确定合适的大小或者使用默认值。
# 3.由于缓存中存储了计算结果，可能导致内存占用过大，需谨慎使用。
# 4.可以是多参数的。
```
### (3)bisect（二分查找）
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
### (4)年份calendar包
```python
import calendar
print(calendar.isleap(2020)) # True
```
### (5)heapq 优先队列
```python
import heapq # 优先队列可以实现以log复杂度拿出最小（大）元素
lst=[1,2,3]
heapq.heapify(lst) # 将lst优先队列化
heapq.heappop(lst) # 从队列中弹出树顶元素（默认最小，相反数调转）
heapq.heappush(lst,element) # 把元素压入堆中
```
### (6)Counter包
```python
from collections import Counter 
# O(n)
# 创建一个待统计的列表
data = ['apple', 'banana', 'apple', 'orange', 'banana', 'apple']
# 使用Counter统计元素出现次数
counter_result = Counter(data) # 返回一个字典类型的东西
# 输出统计结果
print(counter_result) # Counter({'apple': 3, 'banana': 2, 'orange': 1})
print(counter_result["apple"]) # 3
```
### (7)default_dict
defaultdict是Python中collections模块中的一种数据结构，它是一种特殊的字典，可以为字典的值提供默认值。当你使用一个不存在的键访问字典时，defaultdict会自动为该键创建一个默认值，而不会引发KeyError异常。
defaultdict的优势在于它能够简化代码逻辑，特别是在处理字典中的值为可迭代对象的情况下。通过设置一个默认的数据类型，它使得我们不需要在访问字典中不存在的键时手动创建默认值，从而减少了代码的复杂性。
使用defaultdict时，首先需要导入collections模块，然后通过指定一个默认工厂函数来创建一个defaultdict对象。一般来说，这个工厂函数可以是int、list、set等Python的内置数据类型或者自定义函数。
```python
from collections import defaultdict
# 创建一个defaultdict，值的默认工厂函数为int，表示默认值为0
char_count = defaultdict(int)
# 统计字符出现次数
input_string = "hello"
for char in input_string:
    char_count[char] += 1
print(char_count)  # 输出 defaultdict(<class 'int'>, {'h': 1, 'e': 1, 'l': 2, 'o': 1})）
```
### (8)itertools包 (排列组合等)
```python
import itertools
my_list = ['a', 'b', 'c']
permutation_list1 = list(itertools.permutations(my_list))
permutation_list2 = list(itertools.permutations(my_list, 2))
combination_list = list(itertools.combinations(my_list, 2))
bit_combinations = list(itertools.product([0, 1], repeat=4))

print(permutation_list1)
# [('a', 'b', 'c'), ('a', 'c', 'b'), ('b', 'a', 'c'), ('b', 'c', 'a'), ('c', 'a', 'b'), ('c', 'b', 'a')]
print(permutation_list2)
# [('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]
print(combination_list)
# [('a', 'b'), ('a', 'c'), ('b', 'c')]
print(bit_combinations)
# [(0, 0, 0, 0), (0, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 1), (0, 1, 1, 0), (0, 1, 1, 1), (1, 0, 0, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 0, 1, 1), (1, 1, 0, 0), (1, 1, 0, 1), (1, 1, 1, 0), (1, 1, 1, 1)]
```
### (9)正则表达式
```python
import re
match = re.search(pattern, text)#在整个字符串中搜索匹配正则表达式的第一个位置。如果找到匹配的子串，则返回一个匹配对象，否则返回None。
match = re.match(pattern, text)#从字符串的开头开始匹配正则表达式。如果找到匹配的子串，则返回一个匹配对象，否则返回None。
print("YES" if re.search('h.*e.*l.*l.*o',a) else "NO")
```
![[Pasted image 20251223162645.png]]
![[Pasted image 20251223162712.png]]
# 3.算法模板
## (1)贪心
### 最少硬币问题
有1、2、5、10、20、50、100七种面值的硬币，要支付指定的金额，问怎么支付所用的硬币个数最少
**策略：紧着最大分值换。**
### 最大斜率问题
给出n个点的坐标（笛卡尔坐标） 求(A[i]-A[j])/(i-j)最大
**策略：相邻的坐标中找到最大斜率**
### Huffman
* **剪绳子**
```python

import heapq
n=int(input())
a=list(map(int,input().split()))
heapq.heapify(a)
ans=0
for i in range(n-1):
    x=heapq.heappop(a)
    y=heapq.heappop(a)
    heapq.heappush(a,x+y)
    ans+=(x+y)
print(ans)
```
### 区间贪心问题
* **区间选取**（会场安排问题）
给一个大区间l,r然后给你n个区间，最最多多少个区间没有重复部分
**策略：每选一个之后能给后面的留更多的时间（效果：按结束时间排序）**

* **区间选点问题**
n个闭区间[ai,bi]，取尽量少的点，使得每个闭区间内至少有一个点。
**策略：让这个点出现在一个没有点的区间上，尽可能覆盖多的区间的地方（效果：按结束处排序）**
Radar Installation
```python
import functools
from math import sqrt
class isle:
    def __init__(self):
        self.x=0
        self.y=0
class point:
    def __init__(self):
        self.left=0
        self.right=0
def cmp(self,other):
    if(self.right>other.right):
        return 1
    elif(self.right==other.right):
        return 0
    else:
        return -1
case=0
while(True):
    n,d=map(int,input().split())
    case+=1
    flag=1
    ans=[]
    a=[]
    p=[]
    cnt=0
    if(n==0 and d==0):
        break
    for i in range(n):
        t=isle()
        t.x,t.y=map(int,input().split())
        if(t.y>d):
            flag=0
        a.append(t)
        tp=point()
        if(flag==1):
            tp.left=a[i].x-sqrt(d*d-a[i].y*a[i].y)
            tp.right=a[i].x+sqrt(d*d-a[i].y*a[i].y)
            p.append(tp)
    p=sorted(p,key=functools.cmp_to_key(cmp))#从小到大
    if flag==0:
        print("Case %d: -1" %(case))
    elif(flag==1):
        cur=0
        for i in range(n):
            if(i==0):
                cnt+=1
                cur=p[0].right
            else:
                if(p[i].left>cur):
                    cnt+=1
                    cur=p[i].right
        print("Case %d: %d" %(case,cnt))
    input()
```

* **区间完全覆盖问题**
给定一个长度为m的区间（全部闭合），再给出n条线段的起点和终点（注意这里是闭区间），求最少使用多少条线段可以将整个区间完全覆盖
将所有区间化作此区间的区间，剪辑一下（没用的区间删除）
**策略：在能连接区间左边的情况下，找到向右边扩展最长的位置。（效果：按开头排序，开头一样，右边最长的靠前）**
世界杯只因
求完全覆盖区间的最小子区间数。按区间开头排序，往后一直搜到留下空隙的子区间之前，取这些子区间中结尾最大的。  
用二维数组会超内存，可以用数组套元组。
```python
n=int(input())
a=list(map(int,input().split()))
lst=[]
for i in range(n):
    x=max(1,i+1-a[i])
    y=min(n,i+1+a[i])
    lst.append((x,y))
lst.sort()
right=0
idx=0
ans=0
while(right<n):
    maxr=right
    while(idx<n and lst[idx][0]<=right+1):
        maxr=max(maxr,lst[idx][1])
        idx+=1
    right=maxr
    ans+=1
print(ans)

```
* **区间合并问题**
按开头排序然后比较末端
校门外的树又来了
```python
a=[]
l,m=map(int,input().split())
sum=0
for i in range(m):
    x,y=map(int,input().split())
    a.append([x,y])
ans=l+1
res=[]
a=sorted(a,key=lambda t:t[0])
res.append(a[0])
idx=0
for i in range(1,len(a)):
    if(a[i][0]>a[idx][1]):
        res.append(a[i])
        idx+=1
    else:
        a[idx][1]=max(a[idx][1],a[i][1])
for i in range(len(res)):
    ans-=(res[i][1]-res[i][0]+1)
print(ans)

```

### 其他
* **最短的愉悦旋律长度**
```python
n,m=map(int,input().split())
a=list(map(int,input().split()))
cnt=1
s=set()
for i in range(n):
    s.add(a[i])
    if(len(s)==m):
        cnt+=1
        s.clear()
print(cnt)
```
* **购物**
考虑相同硬币数能够最多能够组成1-m的面值。下一次需要选取小于等于m+1面值的硬币，这样可以完全覆盖。一直选到能够覆盖到x面值。
```python
x,n=map(int,input().split())  
a=list(map(int,input().split()))  
a.sort()  
ans=1  
if(n==0):  
    print(-1)  
elif(a[0]!=1):  
    print(-1)  
else:  
    cur=1  
    while(cur<x):  
        for i in range(n-1,-1,-1):  
            if(a[i]<=cur+1):  
                ans+=1  
                cur+=a[i]  
                break  
    print(ans)
```
## (2)排序
### 归并排序求逆序对
```python
ans=0
def merge(l,r):
    global ans
    mid=(l+r)//2
    if(l==r):
        return
    else:
        merge(l,mid)
        merge(mid+1,r)
    i=l
    j=mid+1
    k=l
    while(i<=mid and j<=r):
        if(a[i]<=a[j]):
            b[k]=a[i]
            k+=1
            i+=1
        elif(a[i]>a[j]):
            ans+=(mid-i+1)
            b[k]=a[j]
            k+=1
            j+=1
    while(i<=mid):
        b[k]=a[i]
        k+=1
        i+=1
    while(j<=r):
        b[k]=a[j]
        k+=1
        j+=1
    for i in range(l,r+1):
        a[i]=b[i]
n=int(input())
a=list(map(int,input().split()))
b=[0 for _ in range(n+1)]
merge(0,n-1)
print(ans)

```
### 等价类
排队
```python
n,d=map(int,input().split())  
a=[]  
for i in range(n):  
    t=int(input())  
    a.append(t)  
check=[1]*n  
ans=[]  
maxm=a[0]  
minm=a[0]  
while(sum(check)!=0):  
    i=0  
    qaq=[]  
    maxm=0  
    minm=0  
    while(i<n):  
        if(check[i]==1):  
            if(len(qaq)==0):  
                check[i]=0  
                qaq.append(a[i])  
                maxm=a[i]  
                minm=a[i]  
            else:  
                if(a[i]>=maxm):  
                    maxm=a[i]  
                elif(a[i]<=minm):  
                    minm=a[i]  
                if(a[i]>=maxm-d and a[i]<=minm+d):  
                    qaq.append(a[i])  
                    check[i]=0  
        i+=1  
    qaq.sort()  
    ans.extend(qaq)  
for i in ans:  
    print(i)

```
### 二位列表排序
```python
b=sorted(a,key=lambda x:x[0])#按第一位排序
c=sorted(a,key=lambda x:x[1])#按第二位排序
```
### Class自定义排序
```python
import functools
class patient:
    def __init__(self):
        self.num=""
        self.age=0

def cmp(self,other):
    if(self.age<other.age):
        return 1
    elif(self.age==other.age):
        return 0
    else:
        return -1

p60_new=sorted(p60,key=functools.cmp_to_key(cmp))
```
### heapq自定义排序
```python
import heapq
import functools
class gas:
    def __init__(self):
        self.distance=0
        self.fuel=0
    def __lt__(self,other):
        return self.fuel>other.fuel
#heapq中存储的是gas类型，按照自定义顺序排序
```
### 字典排序
```python
dict = {'b': 2,'a': 1,'c': 3}
sorted_dict = {key: dict[key] for key in sorted(dict)}#按key排序
sorted_dict = sorted(dict.items(), key=lambda x: x[1])#按value排序，返回元组
sorted_dict = sorted(dict.items(), key=itemgetter(0,1))#先按key再按value排序
sorted_dict = sorted(dict.items(), key=lambda x: x[1], reverse=True)#按value排序，倒序
```
## (3)并查集
### 路径压缩+按秩合并
```python
n,m=map(int,input().split())
fa=[0 for _ in range(200005)]
siz=[0 for _ in range(200005)]
def find(x):
    if(x==fa[x]):
        return x
    fa[x]=find(fa[x])
    return fa[x]
def merge(x,y):
    fx=find(x)
    fy=find(y)
    if(fx==fy):
        return
    if(siz[fx]>siz[fy]):
        fa[fy]=fx
        siz[fx]+=siz[fy]
    else:
        fa[fx]=fy
        siz[fy]+=siz[fx]

for i in range(n):
    fa[i]=i
    siz[i]=1
for i in range(m):
    z,x,y=map(int,input().split())
    if(z==1):
        merge(x,y)
    else:
        if(find(x)==find(y)):
            print("Y")
        else:
            print("N")
```
### 种类并查集
* **食物链**
```python
n,k=map(int,input().split())
fa=[0 for _ in range(300005)] #1-n self;n+1-2n prey;2n+1-3n predator
siz=[1 for _ in range(300005)]
def find(x):
    if(x==fa[x]):
        return x
    else:
        fa[x]=find(fa[x])
        return fa[x]
def merge(x,y):
    fx=find(x)
    fy=find(y)
    if(fx==fy):
        return
    if(siz[fx]>siz[fy]):
        fa[fy]=fx
        siz[fx]+=siz[fy]
    else:
        fa[fx]=fy
        siz[fy]+=siz[fx]
for i in range(1,3*n+1):
    fa[i]=i
ans=0
for i in range(k):
    d,x,y=map(int,input().split())
    if(x>n or y>n):
        ans+=1
        continue
    if(d==1):
        if(find(x+n)==find(y) or find(x+2*n)==find(y)):
            ans+=1
            continue
        merge(x,y)
        merge(x+n,y+n)
        merge(x+2*n,y+2*n)
    elif(d==2):
        if(x==y):
            ans+=1
            continue
        if(find(x)==find(y) or find(x)==find(y+2*n)):
            ans+=1
            continue
        merge(x,y+n)
        merge(x+n,y+2*n)
        merge(x+2*n,y)
print(ans)

```
## (4)dfs
### 全排列
```python
def dfs(cur):
    global lena,vis
    if(len(cur)==lena):
        res.append(''.join(cur))
        return
    for i in range(lena):
        if(vis[i]==0):
            vis[i]=1
            cur.append(a[i])
            dfs(cur)
            cur.pop()
            vis[i]=0
a=input()
lena=len(a)
vis=[0 for _ in range(lena)]
res=[]
dfs([])
for i in res:
    print(i)
```
### 全排列（有重复）
同一个位置不放两个相同的数。具体实现方法为每次只取一串相同数中的第一个没有用过的数。注意and与or判断需要加括号以明确优先级。
```python
n=int(input())
a=list(map(int,input().split()))
a.sort()
vis=[0]*n
def dfs(step,cur):
    if(step==n):
        print(" ".join(map(str,cur)))
        return
    for i in range(n):
        if(vis[i]==0 and (i==0 or (a[i]!=a[i-1] or vis[i-1]==1))):
            temp=cur.copy()
            temp.append(a[i])
            vis[i]=1
            dfs(step+1,temp)
            vis[i]=0
dfs(0,[])

```
### 八皇后
```python
n=int(input())
res=[]
#x=[0]*8
y=[0]*8
z=[0]*15
w=[0]*15
def dfs(step,lst):
    if(step==8):
        res.append(lst)
        return
    for i in range(8):
        if(y[i]==0 and z[step+i]==0 and w[7+step-i]==0):
            y[i]=1
            z[step+i]=1
            w[7+step-i]=1
            dfs(step+1,lst+str(i+1))
            y[i]=0
            z[step+i]=0
            w[7+step-i]=0
dfs(0,"")
for i in range(n):
    t=int(input())
    print(res[t-1])
```
### Catalan数枚举
向右或向上走，不能超过对角线。
```python
class Solution:
    def generateParenthesis(self, n: int) -> list[str]:
        ans=[]
        def dfs(l,res,r):
            if(l<r):
                return
            if(l+r==2*n-1):
                ans.append(res+')')
                return
            if(l<n):
                dfs(l+1,res+'(',r)
            if(l>r):
                dfs(l,res+')',r+1)
        dfs(1,"(",0)
        return ans
print(Solution().generateParenthesis((3)))
```
### 其他
* **水淹七军**
```python
import sys
sys.setrecursionlimit(300000)
input=sys.stdin.read
def dfs(x,y,h,a,water,m,n):
    dx=[1,0,-1,0]
    dy=[0,-1,0,1]
    for i in range(4):
        x0=x+dx[i]
        y0=y+dy[i]
        if(0<=x0<m and 0<=y0<n):
            if(a[x0][y0]<h):
                if(water[x0][y0]<h):
                    water[x][y]=h
                    dfs(x0,y0,h,a,water,m,n)

data=input().split()
idx=0
k=int(data[idx])
idx+=1
res=[]
for q in range(k):
    m,n=map(int,data[idx:idx+2])
    idx+=2
    #flag=1
    a=[]
    water=[[0 for _ in range(n)]for _ in range(m)]
    for i in range(m):
        a.append(list(map(int,data[idx:idx+n])))
        idx+=n
    i0,j0=map(int,data[idx:idx+2])
    idx+=2
    i0-=1
    j0-=1
    p=int(data[idx])
    idx+=1
    for i in range(p):
        x,y=map(int,data[idx:idx+2])
        idx+=2
        x-=1
        y-=1
        if(a[x][y]<=a[i0][j0]):
            continue
        dfs(x,y,a[x][y],a,water,m,n)
    if(water[i0][j0]>0):
        print("Yes")
    else:
        print("No")
```
* **数的划分**（剪枝）
```python
n,k=map(int,input().split())  
ans=0  
def dfs(step,cur,s):  
    global ans  
    if(s<0):  
        return  
    if(s==0 and step<k):  
        return  
    if(step==k):  
        if(s==0):  
            ans+=1  
        return  
    for i in range(cur,s//(k-step)+1):  
        dfs(step+1,i,s-i)  
dfs(0,1,n)  
print(ans)

```
## (5)bfs
### 有障碍物
* **逃离紫罗兰监狱**
由于障碍物的存在，使用三维bfs，同时记录穿过障碍物的数量。
```python
from collections import deque
r,c,k=map(int,input().split())
a=[]
vis=[[[0]*(k+1) for _ in range(c)]for _ in range(r)]
q=deque()
dx=[0,1,0,-1]
dy=[1,0,-1,0]
tx,ty,sx,sy=0,0,0,0
ans=-1
def bfs():
    global sx,sy,tx,ty,r,c,k
    q.append((sx,sy,0,0))
    vis[sx][sy][0]=1
    while(q):
        x0,y0,step,cnt=q.popleft()
        for i in range(4):
            x1=x0+dx[i]
            y1=y0+dy[i]
            if(0<=x1<r and 0<=y1<c and vis[x1][y1][cnt]==0):
                vis[x1][y1][cnt]=1
                if(a[x1][y1]=='.'):
                    q.append((x1,y1,step+1,cnt))
                elif(a[x1][y1]=='#'):
                    if(cnt<k):
                        q.append((x1,y1,step+1,cnt+1))
                elif(a[x1][y1]=='E'):
                    return step+1
    return -1
for i in range(r):
    l=input()
    a.append(l)
    if('E' in l):
        tx=i
        ty=l.index('E')
    elif('S' in l):
        sx=i
        sy=l.index('S')
print(bfs())

```
### 0-1bfs、多源bfs
* **愉悦的假期**
从每个岛向全图搜索，记录每个点到三个岛的最短距离。当当前路径大小更小时更新并加入队列。
```python
from collections import deque
n,m=map(int,input().split())
dx=[0,1,0,-1]
dy=[1,0,-1,0]
a=[]
for i in range(n):
    t=input()
    a.append(t)
t=[[0 for _ in range(m)]for _ in range(n)]
def dfs(x,y,mark):
    for i in range(4):
        xx=x+dx[i]
        yy=y+dy[i]
        if(0<=xx<n and 0<=yy<m):
            if(t[xx][yy]==0 and a[xx][yy]=='X'):
                t[xx][yy]=mark
                dfs(xx,yy,mark)
cnt=0
for i in range(n):
    for j in range(m):
        if(a[i][j]=='X' and t[i][j]==0):
            cnt+=1
            t[i][j]=cnt
            dfs(i,j,cnt)
ans=float("inf")
res=[[[float("inf") for _ in range(4)]for _ in range(m)]for _ in range(n)]
def bfs(x):
    q=deque()
    #vis=set()
    for i in range(n):
        for j in range(m):
            if(t[i][j]==x):
                q.append((0,i,j))
                #vis.add((i,j))
                res[i][j][x]=0
    while(q):
        l,i,j=q.popleft()
        for k in range(4):
            xx=i+dx[k]
            yy=j+dy[k]
            if(0<=xx<n and 0<=yy<m):
                #vis.add((xx,yy))
                if(a[xx][yy]=='X'):
                    weight=0
                else:
                    weight=1
                if(l+weight<res[xx][yy][x]):
                    res[xx][yy][x]=l+weight
                    if(weight==0):
                        q.appendleft((l,xx,yy))
                    else:
                        q.append((l+1,xx,yy))
for i in range(1,4):
    bfs(i)
ans=float("inf")
for i in range(n):
    for j in range(m):
        t=res[i][j][1]+res[i][j][2]+res[i][j][3]
        if(a[i][j]=='.'):
            t-=2
        ans=min(ans,t)
print(ans)

```
## (6)Dijkstra
* **Subway**
```python
from math import sqrt
import heapq
def get_dis(x,y):
    a,b=x
    c,d=y
    t=(a-c)**2+(b-d)**2
    return sqrt(t)
x0,y0,x1,y1=map(int,input().split())
stations=[(x0,y0),(x1,y1)]
subway=[]
while(True):
    try:
        t=list(map(int,input().split()))
        if(t==[-1,-1]):
            continue
        qaq=len(t)
        tot=[]
        for i in range(0,qaq-2,2):
            tot.append((t[i],t[i+1]))
            stations.append((t[i],t[i+1]))
        subway.append(tot)
    except EOFError:
        break
n=len(stations)
dis=[[float("inf") for _ in range(n)]for _ in range(n)]
for i in range(n):
    for j in range(i,n):
        if(i==j):
            dis[i][j]=0
        else:
            dis[i][j]=dis[j][i]=get_dis(stations[i],stations[j])/(10/3.6)
for s in subway:
    tmp=len(s)
    for i in range(tmp-1):
        t0=stations.index(s[i])
        t1=stations.index(s[i+1])
        dis[t0][t1]=dis[t1][t0]=get_dis(s[i],s[i+1])/(40/3.6)
h=[(0,0)]
path=[float("inf")]*n
path[0]=0
while(h):
    l,cur=heapq.heappop(h)
    if(l>path[cur]):
        continue
    for i in range(n):
        d=l+dis[i][cur]
        if(d<path[i]):
            path[i]=d
            heapq.heappush(h,(d,i))
print(round(path[1]/60))
```
## (7)dp
### LIS
* **最长上升子序列和**
```python
n=int(input())
a=list(map(int,input().split()))
dp=[1 for _ in range(1005)]#以i结尾的LIS大小
for i in range(n):
    dp[i]=a[i]
for i in range(1,n):
    for j in range(i):
        if(a[j]<a[i]):
            dp[i]=max(dp[i],dp[j]+a[i])
print(max(dp))
```
* **最长上升子序列**
$O(n^2)$
```python
n=int(input())
a=list(map(int,input().split()))
dp=[1 for _ in range(1005)]#以i结尾的LIS长度
for i in range(1,n):
    for j in range(i):
        if(a[j]<a[i]):
            dp[i]=max(dp[i],dp[j]+1)
print(max(dp))
```
$O(nlogn)$ 
* **导弹拦截**
存一个数组记录长度为i的LIS中结尾最小的那个数，通过二分可以快速将数加入数组。贪心可得数组的下标即为LIS长度。  
通过Dilworth定理可知把序列分成不降子序列的最少个数，等于序列的最长下降子序列长度。
```python
def search1(low,l,r,x):
    while(l<=r):
        mid=(l+r)//2
        if(low[mid]<=x):
            l=mid+1
        else:
            r=mid-1
    return l
def search2(low,l,r,x):
    while(l<=r):
        mid=(l+r)//2
        if(low[mid]<x):
            l=mid+1
        else:
            r=mid-1
    return l
def LIS1():
    global lena
    low=[]  # 长度为i的上升子序列结尾的最小值
    low.append(a[0])
    idx=0
    for i in range(1,lena):
        if(a[i]>=low[idx]):
            idx+=1
            low.append(a[i])
        else:
            low[search1(low,0,idx,a[i])]=a[i]
    return idx+1
def LIS2():
    global lena
    low=[]  # 长度为i的上升子序列结尾的最小值
    low.append(a[0])
    idx=0
    for i in range(1,lena):
        if(a[i]>low[idx]):
            idx+=1
            low.append(a[i])
        else:
            low[search2(low,0,idx,a[i])]=a[i]
    return idx+1
a=list(map(int,input().split()))
lena=len(a)
a=a[::-1]
print(LIS1())
a=a[::-1]
print(LIS2())
```
* **分发糖果**
考虑连续上升子序列，此时会对子序列中的每一个人有约束。但是从左往右扫的时候无法以O(n)的复杂度判断下降子序列。因此从右往左再扫一遍。left，right数组分别记录扫描过程中要求的最少糖果数，初值置1.最后每个人的最少糖果数即为两数组中记录的较大值，此时一定满足条件。
### LCS
* **公共子序列**
```python
while True:
    try:
        x,y=input().split()
    except EOFError:
        break
    a=' '+x
    b=' '+y
    lena=len(a)
    lenb=len(b)
    dp=[[0 for _ in range(lenb)] for _ in range(lena)]#a以i结尾，b以j结尾
    ans=0
    for i in range(1,lena):
        for j in range(1,lenb):
            if(a[i]==b[j]):
                dp[i][j]=dp[i-1][j-1]+1
            else:
                dp[i][j]=max(dp[i-1][j],dp[i][j-1])
            ans=max(ans,dp[i][j])
    print(ans)

```
### 区间dp
* **乘积最大**
```python
n,k=map(int,input().split())
l=input()
dp=[[0 for _ in range(50)]for _ in range(50)]#前i个元素插入j个乘号
for i in range(k+1):
    dp[0][i]=0
for i in range(n):
    dp[i][0]=int(l[:i+1])
for i in range(n):
    for j in range(1,k+1):
        for p in range(i):
            dp[i][j]=max(dp[i][j],dp[p][j-1]*int(l[p+1:i+1]))
maxm=0
for i in range(n):
    if(dp[i][k]>=maxm):
        maxm=dp[i][k]
print(maxm)
```
* **统计单词个数**
1.把长度小于单词长度的情况跳过。  
2.先预处理，把第一个位置拥有的单词标注出来。然后按照位置逐个搜索，有存过就加一。  
3.由于字典最多六个单词，使用位运算代替二维数组。第几个单词就在二进制第几位等于一。
```python
def search(left,right):
    global p,s
    ans=0
    for i in range(left,right):
        for j in range(s):
            if(pre[i]&(1<<j) and i+len(dic[j])<=right+1):
                ans+=1
                break
    return ans
p,k=map(int,input().strip().split())
a=""
dic=[]
for i in range(p):
    a+=input().strip()
s=int(input().strip())
for i in range(s):
    dic.append(input().strip())
pre=[0 for _ in range(len(a))]
for j in range(s):
    idx=0
    while(True):
        idx=a.find(dic[j],idx)
        if(idx==-1):
            break
        pre[idx]+=(1<<j)
        idx+=1
dp=[[0 for _ in range(k)] for _ in range(len(a))]#0-i位插j个隔板
for i in range(len(a)):
    dp[i][0]=search(0,i)
for i in range(len(a)):
    for j in range(1,k):
        for q in range(j,i):
            dp[i][j]=max(dp[i][j],dp[q][j-1]+search(q+1,i))
print(dp[len(a)-1][k-1])

```
### 数的划分
n分成k个正整数
```python
n,k=map(int,input().split())  
dp=[[0 for _ in range(k+1)]for _ in range(n+1)]#把i划分为j组的方案数  
for i in range(1,n+1):  
    dp[i][1]=1  
for i in range(1,n+1):  
    for j in range(2,k+1):  
        if(i==j):  
            dp[i][j]=1  
        elif(i>j):  
            dp[i][j]=dp[i-1][j-1]+dp[i-j][j]  
print(dp[n][k])
```
n分成k个自然数
```python
t=int(input())
for i in range(t):
    m,n=map(int,input().split())
    dp=[[0 for _ in range(n+1)]for _ in range(m+1)]
    for i in range(m+1):
        dp[i][0]=dp[i][1]=1
    for i in range(n+1):
        dp[0][i]=dp[1][i]=1
    for i in range(2,m+1):
        for j in range(2,n+1):
            dp[i][j]=dp[i-j][j]+dp[i][j-1]
    print(dp[m][n])
```
### Kadane
* **最大子矩阵**
```python
import sys
n=int(input())
rd=list(map(int,sys.stdin.read().split()))
a=[rd[i*n:(i+1)*n]for i in range(n)]
pre=[[0 for _ in range(n)] for _ in range(n)]
for j in range(n):
    pre[0][j]=a[0][j]
for j in range(n):
    for i in range(1,n):
        pre[i][j]=pre[i-1][j]+a[i][j]#第j列，前i行的和
ans=float('-inf')
dp=[float('-inf') for _ in range(n)]#以i为结尾的最大字段和
dp[0]=pre[0][0]
for j in range(n):#第0-j行(i=0)
    dp[0]=pre[j][0]
    for k in range(1,n):
        dp[k]=max(dp[k-1]+pre[j][k],pre[j][k])
    ans=max(ans,max(dp))
for i in range(1,n):
    for j in range(i,n):#第i-j行
        dp[0]=pre[j][0]-pre[i-1][0]
        for k in range(1,n):
            dp[k]=max(dp[k-1]+pre[j][k]-pre[i-1][k],pre[j][k]-pre[i-1][k])
        ans=max(ans,max(dp))
print(ans)

```
### 01背包
* **采药**
```python
t,m=map(int,input().split())
val=[]
time=[]
dp=[[0 for _ in range(t+1)] for _ in range(m)]#放前i个物品，m的时间
for i in range(m):
    x,y=map(int,input().split())
    time.append(x)
    val.append(y)
for i in range(m):
    for j in range(t,0,-1):
        if(j>=time[i]):
            dp[i][j]=max(dp[i-1][j],dp[i-1][j-time[i]]+val[i])
        else:
            dp[i][j]=dp[i-1][j]
ans=0
for i in range(m):
    ans=max(ans,dp[i][t])
print(ans)
```
### 完全背包
* **Piggy Bank**
```python
t=int(input())
for O_o in range(t):
    e,f=map(int,input().split())
    n=int(input())
    p=[]
    w=[]
    for i in range(n):
        p0,w0=map(int,input().split())
        p.append(p0)
        w.append(w0)
    dp=[float("inf")]*(f-e+1)#容量为i的最低价值
    dp[0]=0
    for i in range(1,f-e+1):
        for j in range(n):
            if(i>=w[j]):
                dp[i]=min(dp[i],dp[i-w[j]]+p[j])
    if(dp[f-e]==float("inf")):
        print("This is impossible.")
    else:
        print("The minimum amount of money in the piggy-bank is %d." %(dp[f-e]))

```
### 状压dp
* **黑神话·悟空**
用17位二进制数存储是否已打败怪兽。dp数组记录打败某些怪兽所需的最少次数。复杂度为O(n∗2n).  
bit_count()用oj过不了。直接用bin().count()计算二进制中1的个数。
```python
a=list(map(int,input().split()))
n=len(a)
MAXM=float("inf")
dp=[MAXM for _ in range(1<<n)]#打掉bitmask中所有boss所需最少任务数
dp[0]=0
for i in range(1<<n):
    gain=bin(i).count('1')+1
    for j in range(n):
        if(not (i>>j)&1):
            t=i|(1<<j)
            dp[t]=min(dp[t],dp[i]+(a[j]+gain-1)//gain)
print(dp[(1<<n)-1])
```
### 最短Hamilton路
* **旅行售货商问题**
状压dp。由于路径是一个环，考虑将起点固定为0号点。dp[i][j]记录从0到j，状态为i时的最短路径。  
枚举状态和目标点j。对每个途径的k点，使用类似Floyd算法对最短路进行更新。  
最终答案即为从0到所有点的最短路加上从这个点回到0点的路径和的最小值。
```python
n=int(input())
a=[]
for i in range(n):
    t=list(map(int,input().split()))
    a.append(t)
dp=[[float("inf") for _ in range(n)]for _ in range(1<<n)]#从起点到j号点，状态为i时的最短路
dp[1][0]=0
for i in range(1<<n):
    if(i&1==0):
        continue
    for j in range(n):
        if((1<<j)&i)==0:
            for k in range(n):
                if((1<<k)&i):
                    dp[i|(1<<j)][j]=min(dp[i|(1<<j)][j],dp[i][k]+a[k][j])
minm=float("inf")
for i in range(n):
    minm=min(minm,dp[(1<<n)-1][i]+a[i][0])
print(minm)

```
### 枚举子集
* **POI 2004 PRZ**
状压dp枚举子集。j=i&(j-1).复杂度O(3^n).  
对每个状态枚举其子集，用剩余的直接过桥时间加子集的dp值更新之。  
注意子集包括0，不要碰到零就停止枚举。
```python
a,n=map(int,input().split())
t0=[]
w0=[]
t=[0]*(1<<n)
w=[0]*(1<<n)
for i in range(n):
    x,y=map(int,input().split())
    t0.append(x)
    w0.append(y)
for i in range(1<<n):
    for j in range(n):
        if((1<<j)&i):
            t[i]=max(t[i],t0[j])
            w[i]+=w0[j]
dp=[float("inf")]*(1<<n)
dp[0]=0
for i in range(1,1<<n):
    j=i
    while(True):
        if(w[i^j]<=a):
            dp[i]=min(dp[i],dp[j]+t[i^j])
        if(j==0):
            break
        j=i&(j-1)
print(dp[(1<<n)-1])

```
### 最长回文子串
```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n=len(s)
        dp=[[0 for _ in range(n)]for _ in range(n)]#i-j是否回文
        for i in range(n):
            dp[i][i]=1
            if(i<n-1 and s[i]==s[i+1]):
                dp[i][i+1]=1
        for j in range(3,n+1):
            for i in range(n-1):
                r=i+j-1
                if(r>=n):
                    break
                dp[i][r]=dp[i+1][r-1]&(s[i]==s[r])
        ans=0
        res=""
        for i in range(n):
            for j in range(i,n):
                if(dp[i][j]==1):
                    if(j-i+1>ans):
                        ans=j-i+1
                        res=s[i:j+1]
        return res
print(Solution().longestPalindrome("aaaaa"))
```
### 双端取数博弈
* **预测赢家**
```python
n=int(input())
for _ in range(n):
    a=list(map(int,input().split()))
    m=a[0]
    nums=a[1:m+1]
    dp=[[0 for _ in range(m)]for _ in range(m)]#i-j,先手比后手多的分数
    for i in range(m):
        dp[i][i]=nums[i]
    for l in range(2,m+1):
        for i in range(m-l+1):
            j=i+l-1
            dp[i][j]=max(nums[i]-dp[i+1][j],nums[j]-dp[i][j-1])
    if(dp[0][m-1]>=0):
        print("true")
    else:
        print("false")

```
### 双dp
* **土豪购物**
```python
a=list(map(int,input().split(',')))
n=len(a)
dp1=a.copy()#不放回
dp2=[0]*n#放回
dp2[0]=a[0]
for i in range(1,n):
    dp1[i]=max(dp1[i-1]+a[i],a[i])
    dp2[i]=max(dp1[i-1],dp2[i-1]+a[i],a[i])
print(max(max(dp1),max(dp2)))

```
### 编辑距离
dp数组记录从word1的0-i位置变到word2的0-j位置需要的最少步数。分$word1[i]$与$word2[j]$是否相等讨论。  
$dp[i-1][j-1]$为替换，$dp[i-1][j]$为插入，$dp[i][j-1]$为删除。边界条件为word1为空或word2为空，因此细节上注意word的1-n对应原字符串的0-n-1.
```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n1=len(word1)#1-n1
        n2=len(word2)
        dp=[[0 for _ in range(n2+1)]for _ in range(n1+1)]#word1i位置换到word2j位置最少步数
        for i in range(1,n2+1):
            dp[0][i]=dp[0][i-1]+1
        for i in range(1,n1+1):
            dp[i][0]=dp[i-1][0]+1
        for i in range(1,n1+1):
            for j in range(1,n2+1):
                if(word1[i-1]==word2[j-1]):
                    dp[i][j]=dp[i-1][j-1]
                else:
                    dp[i][j]=min(dp[i-1][j-1],dp[i-1][j],dp[i][j-1])+1
        return dp[n1][n2]

print(Solution().minDistance("horse","ros"))

```
### 数位dp
* **二进制问题**
```python
n,k=map(int,input().split())
dp=[[[0 for _ in range(2)]for _ in range(72)]for _ in range(72)]#首位为t，含有j个1的i位数个数
dp[1][1][1]=1
dp[1][0][0]=1
for i in range(2,72):
    for j in range(i+1):
        if(j>0):
            dp[i][j][1]=dp[i-1][j-1][1]+dp[i-1][j-1][0]
        dp[i][j][0]=dp[i-1][j][0]+dp[i-1][j][1]
n2=bin(n)[2:]
l=len(n2)
ans=0
qaq=0
for i in range(l):
    if(n2[i]=='1'):
        qaq+=1
if(qaq==k):
    ans=1
for i in range(k,l):
    ans+=dp[i][k][1]
t=1
for i in range(1,l):
    if(t>k):
        break
    if(n2[i]=='1'):
        ans+=dp[l-i][k-t][0]
        t+=1
print(ans)

```
## (8)数据结构
### 栈
* **波兰表达式
```python
def cal(a,b,s):
    if(s=='+'):
        return a+b
    elif(s=='-'):
        return a-b
    elif(s=='*'):
        return a*b
    elif(s=='/'):
        return a/b
a=list(input().split())
sgn=[]
num=[]
check=['+','-','*','/']
for i in range(len(a)-1,-1,-1):
    if(a[i] in check):
        x=num.pop()
        y=num.pop()
        num.append(cal(x,y,a[i]))
    else:
        num.append(float(a[i]))
print("{:.6f}".format(num[0]))
```
* **咒语序列**（最长括号匹配序列）
用栈的思路解决。left,right分别记录左括号和右括号的数目，如果右括号多于左括号就同时置零，如果左=右就更新ans。  
注意会有左括号始终多于右括号的情况，再从右往左扫一遍。
```python
s=input()
left=0
right=0
ans=0
for i in range(len(s)):
    if(s[i]=='('):
        left+=1
    else:
        right+=1
    if(right>left):
        left=0
        right=0
    elif(left==right):
        ans=max(ans,left*2)
s=s[::-1]
left=0
right=0
for i in range(len(s)):
    if(s[i]==')'):
        left+=1
    else:
        right+=1
    if(right>left):
        left=0
        right=0
    elif(left==right):
        ans=max(ans,left*2)
print(ans)

```
### 约瑟夫问题
递推式
```python
while(True):
    n,m=map(int,input().split())
    if(n==0 and m==0):
        break
    a=[]
    for i in range(1,n+1):
        a.append(i)
    idx=0
    for i in range(n-1):
        num=n-i
        idx=(idx+m-1)%num
        del a[idx]
    print(a[0])
```
双端队列
```python
from collections import deque
while(True):
    n,m=map(int,input().split())
    if(n==0 and m==0):
        break
    a=deque()
    for i in range(1,n+1):
        a.append(i)
    while(len(a)>1):
        for i in range(m-1):
            a.append(a.popleft())
        a.popleft()
    print(a.popleft())
```
### 单调栈
* **护林员盖房子 加强版**
```python
from collections import deque
m,n=map(int,input().split())
a=[]
s=[0]*n
l=[-1]*n
r=[n]*n
ans=0
t=deque()
for i in range(m):
    l=list(map(int,input().split()))
    a.append(l)
for i in range(m):
    for j in range(n):
        if(a[i][j]==0):
            s[j]+=1
        else:
            s[j]=0
    t.clear()
    l = [-1]*n
    r = [n]*n
    for j in range(n):
        while(t and s[j]<=s[t[-1]]):
            r[t[-1]]=j
            t.pop()
        l[j]=t[-1] if t else -1
        t.append(j)
    for j in range(n):
        ans=max(ans,s[j]*(r[j]-l[j]-1))
print(ans)

```
* **最小新整数**
```python
from collections import deque
t=int(input())
for i in range(t):
    n,k=input().split()
    k=int(k)
    a=deque()
    m=len(n)
    for j in range(m):
        if(len(a)==0 or k==0):
            a.append(n[j])
        else:
            while(len(a)>0 and a[-1]>n[j] and k>0):
                k-=1
                a.pop()
            a.append(n[j])
    if(k>0):
        for j in range(k):
            a.pop()
    print("".join(a))

```
* **接雨水**
单调栈。建立一个单调递减的栈，当当前元素高于栈顶时就弹出，并计算当前元素与新栈顶之间的雨水，$(i-left-1)*(min(a[left],a[i])-a[q])$
 ```python
from collections import deque
n=int(input())
a=list(map(int,input().split()))
t=deque()
ans=0
for i in range(n):
    if(len(t)==0):
        t.append(i)
    else:
        while(len(t)>0 and a[t[-1]]<a[i]):
            q=t.pop()
            if(len(t)==0):
                break(i-left-1)*(min(a[left],a[i])-a[q])
                
            left=t[-1]
            ans+=(i-left-1)*(min(a[left],a[i])-a[q])
        t.append(i)
print(ans)
 ```
 
### 辅助栈
* **快速堆猪**
辅助栈。加一个栈记录最小值，最小值栈每次添加的即为当前元素与栈顶元素中的较小值。
```python
a=[]
m=[]
while True:
    try:
        s=input().split()
        if(s[0]=='pop'):
            if(len(a)>0):
                a.pop()
                m.pop()
        elif(s[0]=='min'):
            if(len(m)>0):
                print(m[-1])
        else:
            t=int(s[1])
            a.append(t)
            if(len(m)==0):
                m.append(t)
            else:
                m.append(min(m[-1],t))
    except EOFError:
        break
```
### 滑动窗口
* **滑动窗口最大值**
```python
from collections import deque
class Solution:
    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        n=len(nums)
        q=deque()
        ans=[]
        for i in range(k):
            while(q and nums[i]>=nums[q[-1]]):
                q.pop()
            q.append(i)
        ans.append(nums[q[0]])
        for i in range(k,n):
            while(q and nums[i]>=nums[q[-1]]):
                q.pop()
            q.append(i)
            while(q[0]<=i-k):
                q.popleft()
            ans.append(nums[q[0]])
        return ans
print(Solution().maxSlidingWindow([1,3,-1,-3,5,3,6,7],3))
```
* **独特蘑菇**
```python
from collections import defaultdict
n,k=map(int,input().split())
c=list(map(int,input().split()))
f=defaultdict(int)
ans=0
l=0
for r in range(n):
    if(c[r] not in f):
        f[c[r]]=1
    else:
        f[c[r]]+=1
    while(len(f)>k):
        f[c[l]]-=1
        if(f[c[l]]==0):
           del f[c[l]]
        l+=1
    ans+=(r-l+1)
print(ans)
```
### 双堆+懒删除
* **候选人追踪**
双堆+懒删除。开两个堆分别记录目标中的最小值和非目标中的最大值，等pop的时候判断是否为旧数据再删除。注意特判k=m的情况，此时大根堆应始终返回-1而不是0.
```python
import heapq  
n,k=map(int,input().split())  
qaq=list(map(int,input().split()))  
s=list(map(int,input().split()))  
votes=[]  
for i in range(n):  
    votes.append([qaq[2*i],qaq[2*i+1]])  
votes=sorted(votes,key=lambda x:x[0])  
win=[]  
lose=[]  
cand=set(s)  
m=314159  
cnt=[0]*(m+1)  
for i in range(k):  
    heapq.heappush(win,(0,s[i]))  
cur=0  
lst=0  
ans=0  
def get_min():  
    while(win):  
        num,id=win[0]  
        if(cnt[id]==num):  
            return num  
        heapq.heappop(win)  
    return 0  
def get_max():  
    while(lose):  
        num,id=lose[0]  
        if(cnt[id]==-num):  
            return -num  
        heapq.heappop(lose)  
    if(k==m):  
        return -1  
    else:  
        return 0  
for i in range(n):  
    cur=votes[i][0]  
    win0=get_min()  
    lose0=get_max()  
    if(win0>lose0):  
        ans+=(cur-lst)  
    lst=cur  
    idx=votes[i][1]  
    cnt[idx]+=1  
    if(idx in cand):  
        heapq.heappush(win,(cnt[idx],idx))  
    else:  
        heapq.heappush(lose,(-cnt[idx],idx))  
print(ans)
```
## (9)字符串
### 字典树Trie
```python
class TrieNode:
    def __init__(self):
        self.children={}
        self.is_end=False
class Trie:
    def __init__(self):
        self.root=TrieNode()
    def insert(self,word):
        node=self.root
        for digit in word:
            if digit not in node.children:
                node.children[digit]=TrieNode()
            node=node.children[digit]
            if(node.is_end):
                return False
        node.is_end=True
        return len(node.children)==0
t=int(input())
for i in range(t):
    n=int(input())
    nums=[]
    for j in range(n):
        q=input()
        nums.append(q)
    nums.sort()
    trie=Trie()
    flag=1
    for j in nums:
        if(not trie.insert(j)):
            flag=0
            break
    if(flag==1):
        print("YES")
    else:
        print("NO")
```
### Manacher
```python
temp=input()
s='#'.join('^{}$'.format(temp))
n=len(s)
p=[0]*n#以s[i]为中心的最长回文半径
c=0#当前中心
r=0#当前中心的最右边界
for i in range(1,n-1):
    j=2*c-i
    if(r>i):
        p[i]=min(r-i,p[j])
    else:
        p[i]=0
    while(s[i+(1+p[i])]==s[i-(1+p[i])]):
        p[i]+=1
    if(i+p[i]>r):
        c=i
        r=i+p[i]
print(max(p))
```
### KMP
```python
a="_"+input().strip()
b="_"+input().strip()
l1=len(a)
l2=len(b)
j=0
kmp=[0]*l2
for i in range(2,l2):
    while(j>0 and b[i]!=b[j+1]):
        j=kmp[j]
    if(b[j+1]==b[i]):
        j+=1
    kmp[i]=j
j=0
for i in range(1,l1):
    while(j>0 and b[j+1]!=a[i]):
        j=kmp[j]
    if(b[j+1]==a[i]):
        j+=1
    if(j==l2-1):
        print(i-l2+2)
        j=kmp[j]
for i in range(1,l2):
    print(kmp[i],end=" ")
```
## (10)数学
### 埃氏筛
```python
from math import sqrt
p=[True]*10001
def is_prime(x):
    for i in range(2,int(sqrt(x)+1)):
        if(x%i==0):
            return False
    return True

for i in range(2,101):
    if(is_prime(i)):
        for j in range(i*i,10001,i):
            p[j]=False
```
### 欧拉筛
```python
def sieve_of_euler(x):  
    for i in range(2,x+1):  
        if(is_prime[i]==1):  
            primes.append(i)  
        for p in primes:  
            k=i*p  
            if(k>n):  
                break  
            else:  
                is_prime[k]=0  
            if(i%p==0):  
                break  
n=int(input())  
is_prime=[1 for _ in range(10005)]  
is_prime[0]=is_prime[1]=0  
primes=[]  
sieve_of_euler(n)  
if(n in primes):  
    print("YES")  
else:  
    print("NO")
```
* **Two Divisors**（欧拉筛求最小质因数）
```python
from math import sqrt
n=int(input())
a=list(map(int,input().split()))
b=[]
c=[]
primes=[]
MAXA=max(a)+1
#sieve_of_euler
div=[0]*MAXA
for i in range(2,MAXA):
    if(div[i]==0):
        div[i]=i
        primes.append(i)
    for j in primes:
        if(j*i>=MAXA):
            break
        div[j*i]=j
        if(i%j==0):
            break
```
### 快速幂
把幂次转换为二进制，base定为a的一次方。每次b右移一位并与1按位与，相当于判断b的最后一位是否为1.然后如果为一就乘base，每次操作base都自乘。
* **麦森数**
```python
import math
mod=10**500
def quickpower(a,b):
    ans=1
    base=a
    while(b>0):
        if(b&1):
            ans*=base
        base*=base
        b>>=1
        ans%=mod
    return ans
p=int(input())
print(int(math.log(2,10)*p)+1)
ans=1
t=quickpower(2,p)-1
t=str(t)
if(len(t)<500):
    t='0'*(500-len(t))+t
for i in range(10):
    print(t[i*50:(i+1)*50])

```
### 矩阵快速幂
```python
mod=10**9+7
def mul(x,y,p,q,r):
    qaq=[[0 for _ in range(q)]for _ in range(p)]
    for i in range(p):
        for j in range(q):
            for k in range(r):
                qaq[i][j]+=x[i][k]*y[k][j]%mod
                qaq[i][j]%=mod
    return qaq
def quick_power(x,k):
    global n
    t=[[0 for _ in range(n)]for _ in range(n)]
    for i in range(n):
        t[i][i]=1
    while(k):
        if(k&1):
            t=mul(t,x,n,n,n)
        x=mul(x,x,n,n,n)
        k>>=1
    return t
n,k=map(int,input().split())
a=[]
for i in range(n):
    lst=list(map(int,input().split()))
    a.append(lst)
ans=quick_power(a,k)
for i in range(n):
    print(" ".join(map(str,ans[i])))

```
斐波那契数列-大数据版
```python
mod=10**9+7
def mul(x,y,p,q,r):
    qaq=[[0 for _ in range(q)]for _ in range(p)]
    for i in range(p):
        for j in range(q):
            for k in range(r):
                qaq[i][j]+=x[i][k]*y[k][j]%mod
                qaq[i][j]%=mod
    return qaq
def quick_power(x,n):
    t=[[1,0],[0,1]]
    while(n):
        if(n&1):
            t=mul(t,x,2,2,2)
        x=mul(x,x,2,2,2)
        n>>=1
    return t
n=int(input())
a=[[1,1]]
base=[[1,1],[1,0]]
if(n==1):
    print(1)
else:
    ans=mul(a,quick_power(base,n-2),1,2,2)
    print(ans[0][0])
```
### Catalan数
```python
from math import comb
n=int(input())
a=1
# if(n==1):
#     print(1)
# else:
#     print(comb(2*n,n)//(n+1))
for i in range(2,n+1):
    a=(4*i-2)*a//(i+1)
print(a)
```
### 康托展开
* **排列**
```python
from math import factorial
import heapq
m=int(input())
for p in range(m):
    n,k=map(int,input().split())
    a=list(map(int,input().split()))
    ans=[]
    t=0
    count=[0]*n
    for i in range(n):
        for j in range(i,n):
            if(a[j]<a[i]):
                count[i]+=1
    for i in range(n):
        t+=factorial(n-1-i)*count[i]
    t+=k
    t=(t+1)%(factorial(n))-1
    for i in range(n):
        temp=t//factorial(n-1-i)
        ans.append(temp)
        t=t-temp*factorial(n-1-i)
    res=[]
    nums=[]
    heapq.heapify(nums)
    for i in range(1,n+1):
        nums.append(i)
    for i in range(n):
        qwq=nums[ans[i]]
        res.append(qwq)
        nums.remove(qwq)
    print(" ".join(map(str,res)))
```
### 拉格朗日四平方和定理
2：自然数n可以表为两个平方数之和等价于n的每个形如p=4m+3的素因子的次数为偶数。  
4：4a∗(8k+7)  
其余均能表示为三平方和。
## (11)其他
### 双指针
* **移动零**
左指针指向当前已经处理好的序列的尾部，右指针指向待处理序列的头部。右指针不断向右移动，每次右指针指向非零数，则将左右指针对应的数交换，同时左指针右移。左指针左边均为非零数；右指针左边直到左指针处均为零。因此每次交换，都是将左指针的零与右指针的非零数交换，且非零数的相对顺序并未改变。
```python
class Solution:
    def moveZeroes(self, nums: list[int]) -> None:
        i0=0
        for i in range(len(nums)):
            if(nums[i]!=0):
                nums[i0],nums[i]=nums[i],nums[i0]
                i0+=1
        return nums
```

### 二分查找
```python
class Solution:
    def searchInsert(self, nums: list[int], target: int) -> int:
        left=0
        right=len(nums)-1
        while(left<=right):
            mid=(left+right)//2
            if(nums[mid]>=target):
                right=mid-1
            else:
                left=mid+1
        return left
```
### 二维前缀和
* **护林员盖房子**
```python
m,n=map(int,input().split())
a=[]
for i in range(m):
    t=list(map(int,input().split()))
    a.append(t)
pre=[[0 for _ in range(n+1)]for _ in range(m+1)]#向右向下分别移动一位
pre[1][1]=a[0][0]
for i in range(1,m+1):
    for j in range(1,n+1):
        pre[i][j]=pre[i-1][j]+pre[i][j-1]-pre[i-1][j-1]+a[i-1][j-1]
ans=0
for i in range(1,m+1):
    for j in range(1,n+1):
        for p in range(i,m+1):
            for q in range(j,n+1):
                if(pre[p][q]-pre[i-1][q]-pre[p][j-1]+pre[i-1][j-1]==0):
                    ans=max(ans,(p-i+1)*(q-j+1))
print(ans)
```
### 悬线法
护林员盖房子加强版
```python
m,n=map(int,input().split())
a=[]
l=[0]*(n+1)
r=[0]*(n+1)
s=[0]*(n+1)
ans=0
for i in range(m):
    l=list(map(int,input().split()))
    a.append(l)
for i in range(m):
    for j in range(n):
        if(a[i][j]==0):
            s[j]+=1
        else:
            s[j]=0
    for j in range(n):
        l[j]=r[j]=j
    for j in range(n):
        while(l[j]>0 and s[l[j]-1]>=s[j]):
            l[j]=l[l[j]-1]
    for j in range(n,-1,-1):
        while(r[j]<n-1 and s[r[j]+1]>=s[j]):
            r[j]=r[r[j]+1]
    for j in range(n):
        ans=max(ans,s[j]*(r[j]-l[j]+1))
print(ans)
```
### 差分数组、扫描线
给出若干区间，输出每个位置被覆盖的次数。
差分写法
```python
n = 7
intervals = [(1,3), (2,5), (4,6)]

diff = [0]*(n+2)
for l, r in intervals:
    diff[l] += 1
    diff[r+1] -= 1

cover = [0]*(n+1)
cover[0] = diff[0]
for i in range(1, n+1):
    cover[i] = cover[i-1] + diff[i]

print(cover[1:])  # [1,2,2,2,1,1,0]
```
扫描线写法
```python
events = []
for l, r in intervals:
    events.append((l, 1))
    events.append((r + 1, -1))

events.sort()
res = [0]*(n+1)
cur = 0
idx = 0
for i in range(1, n+1):
    while idx < len(events) and events[idx][0] == i:
        cur += events[idx][1]
        idx += 1
    res[i] = cur

print(res[1:])  # [1,2,2,2,1,1,0]
```
### 后悔解法
* **Potions**
维护一个小根堆。不管怎样先把药水喝了，如果健康值为负值就把堆顶的药水依次吐出来。  
若dp则记录喝i瓶水的剩余生命值。
```python
import heapq
n=int(input())
a=list(map(int,input().split()))
t=[]
sum=0
heapq.heapify(t)
for i in range(n):
    sum+=a[i]
    heapq.heappush(t,a[i])
    while(sum<0 and len(t)>0):
        x=heapq.heappop(t)
        sum-=x
print(len(t))
```
### 模拟
* **Saruman's Army**
两个while循环，一个找开始的点，另一个找放置点。找完一轮之后开始新的一轮。
```css
while(True):
    r,n=map(int,input().split())
    if(r==-1 and n==-1):
        break
    a=list(map(int,input().split()))
    a.sort()
    i=0
    ans=0
    while(i<n):
        start=a[i]
        i+=1
        while(i<n and a[i]<=start+r):
            i+=1
        cur=a[i-1]
        while(i<n and a[i]<=cur+r):
            i+=1
        ans+=1
    print(ans)
```
* **垃圾炸弹**
在接收数据时更新每个点处放置炸弹时可以清除的垃圾数。在range()中使用min()、max()以避免分类讨论。注意边界处的方形边长可以小于2d。
```python
d=int(input())
n=int(input())
a=[[0 for _ in range(1030)]for _ in range(1030)]
for i in range(n):
    x,y,z=map(int,input().split())
    for j in range(max(0,x-d),min(1024,x+d)+1):
        for k in range(max(0,y-d),min(1024,y+d)+1):
            a[j][k]+=z
res=0
ans=0
for i in range(1025):
    for j in range(1025):
        if(a[i][j]>ans):
            res=1
            ans=a[i][j]
        elif(a[i][j]==ans):
            res+=1
print(res,ans)
```
### 汉诺塔问题
```python
def hanoi(x,a,b,c):
    if(x==0):
        return
    hanoi(x-1,a,c,b)
    print("%d:%s->%s" %(x,a,c))
    hanoi(x-1,b,a,c)
n,a,b,c=input().split()
n=int(n)
hanoi(n,a,b,c)
```
### 二分答案
* **袋子里最少数目的球**
```python
class Solution:
    def minimumSize(self, nums: list[int], maxOperations: int) -> int:
        l,r,ans=1,max(nums),0
        while(l<=r):
            mid=(l+r)>>1
            res=0
            for i in nums:
                res+=(i-1)//mid
            if(res<=maxOperations):
                ans=mid
                r=mid-1
            else:
                l=mid+1
        return ans
```