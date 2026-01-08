**大纲**

**1.特殊算法**

1.1.Kadana算法——数组中和最大的子数组

1.2.有分割数限制的数组分割问题

​	1.2.1.最小化子数组和的最大值

​	1.2.2.最小化各子数组的极差之和

1.3矩阵快速幂

​	1.3.1.计算斐波那契数列

​	1.3.2.神经网络之国（+卷积）

1.4归并排序

1.5多数之和

​	1.5.3.三数之和

​	1.5.4四数之和

1.6区间问题

​	1.6.1.区间合并

​	1.6.2.选择最多不相交区间

​	1.6.3.选覆盖全部区间的最少的点

​	1.6.4将区间分为互不相交的组

​	1.6.5.覆盖连续区间（购物）

**2.数据结构**

2.1.并查集

2.2堆

2.3队列

2.4树状数组

2.5字典树

2.6单调栈

2.7辅助栈

**3.动态规划**

3.1.01背包

3.2完全背包

3.3.最优双序列匹配

3.4.最长递减子数组

3.5乘积最大的子数组

3.6两端取数博弈问题

3.7状态压缩dp

**4.搜索**

4.1.深度优先搜索

4.2.广度优先搜索

4.3.Dijkstra

**5.数学相关**

5.1.欧式筛法

5.2.分解因数

5.3.康托展开

5.4.Catalan数

5.5数的拆分问题

​	5.5.1.全部互异分拆/奇数分拆

​	5.5.2.全部分拆

​	5.5.3.划分为k个正整数的分拆

​	5.5.4.划分为k个非负整数（含0）的分拆

**6.内置函数及用法**



**正文**

**1.特殊算法**

1.1.Kadana算法——数组中和最大的子数组

```python
def kadane(arr):
    max_end_here=max_so_far=arr[0]
    for x in arr[1:]:
        max_end_here=max(x,max_end_here+x)
        max_so_far=max(max_so_far,max_end_here)
        return max_so_far
```

1.2.有分割数限制的数组分割问题

​	1.2.1.最小化子数组和的最大值

```python
n,m=map(int,input().split())
expend=[]
for _ in range(n):
    expend.append(int(input()))
def check(x):
    num,s=1,0
    for i in range(n):
        if s+expend[i]>x:
            s=expend[i]
            num+=1
        else:
            s++expend[i]
    return True if num<=m else False
#以下的二分查找法也可用于最大化最小值问题，只需把*1，*2行交换
lo=max(expend)
hi=sum(expend)+1
ans=1
while lo<hi:
	mid=(lo+hi)//2
    if check(mid):
        ans=mid
        hi=mid
    else:
        lo=mid+1
print(ans)
```

​	1.2.2.最小化各子数组的极差之和

```python
import heapq
n,m=map(int,input().split())
arr=list(map(int,input().split()))
arr.sort()
gap=[]
for i in range(1,n):
    gap.append(arr[i-1]-arr[i])
heapq.heapify(gap)
ans=arr[-1]-arr[0]
for _ in range(m-1):
    ans+=heapq.heappop(gap)
print(ans)
```

1.3矩阵快速幂

​	1.3.1.计算斐波那契数列

```python
MOD=10**9+7
def matrix_mult(A,B):
    return[
        [(A[0][0]*B[0][0]+A[0][1]*B[1][0])%MOD,
        (A[0][0]*B[0][1]+A[0][1]*B[1][1])%MOD],
        [(A[1][0]*B[0][0]+A[1][1]*B[1][0])%MOD,
        (A[1][0]*B[0][1]+A[1][1]*B[1][1])%MOD]
    ]
def matrix_pow(matrix,n):
    result=[[1,0],[0,1]]
    base=matrix
    while n>0:
        if n%2==1:
            result=matrix_mult(result,base)
        base=matrix_mult(base,base)
        n//=2
    return result
def fibanacci(n):
    if n==1 or n==2:
        return 1
    F=[[1,1],[1,0]]
    result=matrix_pow(F,n-1)
    return result[0,0]
```

​	1.3.2.神经网络之国（+卷积）http://cs101.openjudge.cn/practice/29741/

```python
MOD=10**9+7
import sys
data=list(map(int,sys.stdin.read().split()))
idx=0
N=data[idx]; idx+=1
L=data[idx]; idx+=1
M=data[idx]; idx+=1
sta=data[idx:idx+N]; idx+=N
mid=data[idx:idx+N]; idx+=N
end=data[idx:idx+N]

def convolution(a,b,m):
    c=[0]*m
    for i in range(m):
        for j in range(m):
            c[(i+j)%m]=(c[(i+j)%m]+a[i]*b[j])%MOD
    return c

def fast_power(f,t,m):
    re=[0]*m
    re[0]=1
    while t>0:
        if t&1==1:
            re=convolution(re,f,m)
        f=convolution(f,f,m)
        t>>=1
    return re

m_sta=[0]*M
for i in sta:
    m_sta[i%M]=(m_sta[i%M]+1)%MOD
m_mid=[0]*M
for i in mid:
    m_mid[i%M]=(m_mid[i%M]+1)%MOD
m_end=[0]*M
for i in range(N):
    m_end[(mid[i]+end[i])%M]=(m_end[(mid[i]+end[i])%M]+1)%MOD

m_mid=fast_power(m_mid,L-2,M)
ans=convolution(m_sta,m_mid,M)
ans=convolution(ans,m_end,M)
print(ans[0])
```

1.4归并排序

```python
def merge_sort(arr):
    if len(arr)<=1:
        return arr
    mid=len(arr)//2
    left=merge_sort(arr[:mid])
    right=merge_sort(arr[mid:])
    return merge(lefft,right)
def merge(left,right):
    result=[]
    while i<len(left) and j<len(right):
        if left[i]<right[j]:
            result.append(left[i])
            i+=1
        else:
            result.append(right[j])
            j+=1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

1.5多数之和

​	1.5.3.三数之和

```python
arr=list(map(int,input().split()))
arr.sort()
ans=set()
for i in range(len(arr)-2):
    if i>0 and a[i]==a[i-1]:
        continue
    d=set()
    for x in arr[i+1:]:
        if x not in d:
            d.add(-x-arr[i])
        else:
            ans.add((arr[i],-x-arr[i],x))
print(len(ans))
```

​	1.5.4四数之和

```python
def sum4(n,A,B,C,D):
    sum_ab={}
    for a in A:
        for b in B:
            s=a+b
            sum_ab[s]=sum_ab.get(s,0)+1
    count=0
    for c in C:
        for d in D:
            s=c+d
            count+=sum_ab.get(-s,0)
    return count
```

1.6区间问题

​	1.6.1.区间合并

```python
def merge(intervals):
    intervals.sort(key=lambda x:x[0])
    ans=[]
    if intervals:
        start,end=intervals[0][0],intervals[0][1]
    	for interval in intervals[1:]:
        	if interval[0]<=end:
                end=max(end,interval[1])
            else:
                ans.append([start,end])
                start=interval[0]
                end=interval[1]
        ans.append([start,end])
return ans
```

​	1.6.2.选择最多不相交区间

```python
import sys
def over_lap_intervals(intervals):
    intervals.sort(key=lambda x:x[1])
    res=0
    end=-sys.maxsize
    for v in intervals:
        if end<=v[0]:
            res+=1
            end=v[1]
    return len(intervals)-res
```

​	1.6.3.选覆盖全部区间的最少的点

```python
import sys
def min_shots(points):
    points.sort(key=lambda x:x[1])
    ans=0
    end=-sys.maxsize
    for p in points:
        if p[0]>end:
            ans+=1
            end=p[1]
    return ans
```

​	1.6.4将区间分为互不相交的组

```python
import heapq
def min_host(start_end):
    start_end.sort(key=lambda x:x[0])
    q=[]
    for i in start_end:
        if not q or q[0]>i[0]:
            heapq.heappush(q,i[1])
        else:
            heapq.heappop(q)
            heapq.heappush(q,i[1])
    return len(q)
```

​	1.6.5.覆盖连续区间（购物）http://cs101.openjudge.cn/practice/29896/

```python
x,n=map(int,input().split())
coins=list(map(int,input().split()))
if x>0 and 1 not in set(coins):
    print(-1)
elif x<=0:
    print(0)
else:
    coins.sort()
    recent=0
    i=0
    ans=0
    while recent<x:
        recent+=coins[i]
        ans+=1
        while i<n-1 and recent>=coins[i+1]-1:
            i+=1
    print(ans)
```

**2.数据结构**

2.1.并查集

```python
class UnionFind:
    def __init__(self,n):
        self.aprent=list(range(n))
        self.rank=[0]*n
    def find(self,x):
        if self.parent[x]!=x:
            self.parent[x]=self.find(self.parent[x])
        return self.parent[x]
    def union(self,x,y):
        root_x=self.find(x)
        root_y=self.find(y)
        if root_x!=root_y:
            if self.rank[root_x]<self.rank[root_y]:
                self.parent[root_x]=root_y
            elif self.rank[root_x]>self.rank[root_y]:
                self.parent[root_y]=root_x
            else:
                self.parent[root_y]=root_x
                self.rank[root_x]+=1
```

2.2堆 

```python
import heapq
heap=list()
item=0
heapq.heapify(heap)#将列表转化为堆
heapq.heappush(heap,item)#插入元素
heapq.heappop(heap)#删除并返回最小元素
heapq.heappushpop(heap,item)#先推入元素，再弹出最小元素
heapq.heapreplace(heap,item)#先弹出最小元素，再推入新元素
```

2.3队列

```python
from collecions import deque
queue=deque(list())
queue.append()
queue.pop()
queue.popleft()
queue.appendleft()
queue.rotate() #旋转，正数右转，负数左转
```

2.4树状数组 

```python
tree=[0]*(n+1)#tree[i]存储了闭区间[i-lowbit(i)+1,i]的和
def lowbit(x):
    return x&-x
def update(pos,delta):
    while pos<=n:
        tree[pos]+=delta
        pos+=lowbit(pos)
def query(pos):
    res=0
    while pos>0:
        res+=tree[pos]
        pos-=lowbit(pos)
    return res
```

2.5字典树 

```python
class TrieNode:
    def __init__(self):
        self.children={}
        self.is_end=False
class Trie:
    def __init__(self)：#插入单词
    	self.rot=TrieNode()
    def insert(self,word:str):
        node=self.root
        for char in word:
            if char not in node.children:
                node,children[char]=TrieNode()
            node=node.children[char]
        node.is_end=True
    def search(self,word:str):#搜索完整单词
        node=self.root
        for char in word:
            if char not in node.children:
                return False
            node=node.children[char]
        return node.is_end
    def startwith(self,pre):#检查是否有以pre开头的单词
        node=self.root
        for char in pre:
            if char not in node.children:
                return False
            node=node.children[char]
        return True
    def delete(self,word:str):#删除单词
        def _delete(node,word.depth):
        	if depth==len(word):
                if not node.is_end:
                    return False
                node.is_end=False
                return len(node.children)==0
            char=word[depth]
            if char not in node.children:
                return False
            should_delete_child=_delete(node.children[char],word,depth+1)
            if should_delete_child:
                del node.children[char]
                return len(node.children)==0 and not node.is_end
            return False
        return _delete(self.root,word,0)
```

2.6单调栈

```python
#一维数组最大矩形面积
def m_stack(heights):
	mx=0
	stack=[]
	for j in range(len(heights)+1):
    	h=heights[j] if j<len(heights) else -1
    	while stack and heights[stack[-1]]>h:
        	height=heights[stack.pop()]
        	width=j if not stack else j-stack[-1]-1
        	mx=max(mx,height*width)
        satck.append(j)
    return mx
```

```python
#应用：01矩阵中的最大矩形面积
row,col=map(int,input().split())
forest=[list(map(int,input().split())) for _ in range(row)]
max_area=0
for i in range(row):
    for j in range(col):
        if forest[i][j]==0:
            heights[j]+=1
        else:
            heights[j]=0
    max_area=max(max_area,m_stack(heights))
print(max_area)
```

2.7辅助栈

```python
#应用1：最小栈
class MinStack:
    def __init__(self):
        self.stack=[]
        self.min_stack=[]
    def push(self,n):
        self.stck.append(n)
        if not self.min_stack or n<=self.min_satck[-1]:
            self.min_stack.append(n)
        else:
            self.min_stack.append(self.min_stack[-1])
    def get_min(self):
        return self.min_stack[-1]
    def pop(self):
        if self.stack:
            self.stack.pop()
            self.min_satck.pop()
```

```python
#应用2：找到数组中的下一个更大元素
def get_greater_element(nums):
    result=[-1]*len(nums)
    stack=[]
    for i in range(len(nums)):
        while stack and nums[i]>nums[stack[-1]]:
            idx=stack.pop()
            result[idx]=nums[i]
        stack.append(i)
    return result
```

**3.动态规划**

3.1.01背包

```python
T,m=map(int,input().split())
dp=[0]*[T+1]
for _ in range(m):
    t,v=map(int,input().split())
    for j in range(T,t-1,-1):
        dp[j]=max(dp[j],dp[j-t]+v)
print(dp[T])
```

3.2完全背包

```python
def min_coins_for_change(amount,coins):
    dp=[float('inf')]*(amount+1)
    for i in range(1,amount+1):
        for coin in coins:
            if i>=coin:
                dp[i]=min(dp[i],dp[i-coin]+1)
    return dp[amount] if dp[amount]!=float('inf') else -1
```

3.3.最优双序列匹配

```python
a,b=input().split()
alne,blen=len(a),leb(b)
dp=[[0]*(blen+1) for _ in range(alen+1)]
for i in range(1,blen+1):
    if a[i-1]==b[j-1]:
        dp[i][j]=dp[i-1][j-1]+1
    else:
        dp[i][j]=max(dp[i-1][j],dp[i][j-1])
print(dp[alen][blen])
```

3.4.最长递减子数组

```python
n=int(input())
arr=list(map(int,input().split()))
dp=[1]*n
for i in range(1,n):
    for j in range(i):
        if arr[j]>=arr[i]:
            dp[i]=max(dp[i],dp[j]+1)
print(max(dp))   
```

3.5乘积最大的子数组(数组中有正数、负数、0)

```python
def max_product(nums):
    re=nums[0]
    p_max=p_min=nums[0]
    for num in nums[1:]:
        c_max=max(p_max*n,p_min*n,n)
        c_min=min(p_max*n,p_min*n,n)
        re=max(re,c_max)
        p_max,p_min=c_max,c_min
    return res
```

3.6两端取数博弈问题

```python
data=list(map(int,input().split()))
m=data[0]
arr=deta[1:]
dp=[[0]*m for _ in range(m)]
for i in range(m):
    dp[i][i]=arr[i]
for k in range(1,m):
    for i in range(m):
        if i+k<m:
            dp[i][i+k]=max(arr[i]-dp[i+1][i+k],arr[i+k]-dp[i][i+k-1])
if dp[o][m-1]>=0:
    print('true')
else:
    print('false')
```

3.7状态压缩dp

经典例子：旅行商问题（TSP） 

问题描述：给定一系列城市和每对城市之间的距离，求访问每个城市恰好一次并回到起点的最短路径。 

状态压缩 DP 解法： 

定义状态 dp[mask][i}：表示已访问城市集合为 mask（二进制数），当前位于城市 i 时的最短路径长度。 

```python
def s_c_dp(n,cost):
    dp=[[float('inf')]*n for _ in range(1<<n)]
    for i in range(n):
        dp[1<<i][i]=0
    for mask in range(1<<n):
        for i in range(n):
            if not (mask&(1<<i)) or dp[mask][i]==float('inf'):
                continue
        for j in range(n):
            if mask&(1<<j):
                continue
            new_mask=mask|(1<<j)
            dp[new_mask][j]=min(dp[new_mask][i],dp[mask][i]+cost[i][j])
    result=floot('inf')
    full_mask=(1<<n)-1
    for i in range(n):
        result=min(result,dp[full_mask][i]+结束代价)
    return result
```

**4.搜索**

4.1.深度优先搜索

```python
def dfs(graph,start):
    visited=[]
    stack=[start]
    while stack:
        node=stack.pop()
        if node not in visited:
            visited.append(node)
            if node in graph:
                for neighbor in reversed(graph[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)
    return visited
```

4.2.广度优先搜索

```python
from collections import deque
def bfs(start,target,graph):
    queue=deque()
    visited=set()
    queue.append((start,0))
    visited.add(start)
    while queue:
        current,distance=queue.popleft()
        if current==target:
            return distance
        for nrighbor in graph[current]:
            if neighbor not in visiteed:
                visited,add(neighbor)
                queue.append((neighbor,distance+1))
```

4.3.Dijkstra

```python
import heapq
import sys
def dijstra(r,c,matrix):
    di=[(1,0),(0,1),(-1,0),(0,-1)]
    dist=[[sys.maxsize]*c for _ in range(r)]
    dist[0][0]=0
    heap=[[0,0,0]]
    while heap:
        e,x,y=heapq.heappop(heap)
        if e>dist[x][y]:
            continue
        if x==r-1 and y==c-1:
            return e
        for dx,dy in di:
            a,b=x+dx,y+dy
            if 0<=a<r and 0<=b<c:
                new_e=max(e,abs(matrix[a][b]-matrix[x][y]))
                if new_e<dist[a][b]:
                    dist[a][b]=new_e
                    heap.heappush(heap,(new_e,a,b))
	return dist[-1][-1]
```

**5.数字相关** 

5.1.欧式筛法

```python
def euler_sieve(n):
    is_prime=[True]*(n+1)
    if_prime[0]=is_prime[1]=False
    primes=[]
    for i in range(2,n+1):
        if is_prime[i]:
            primes.append(i)
        for p in ptimes:
            if i*p>n:
                break
            is_primes[i*p]=False
            if i%p==0:
                break
    return primes
```

5.2.分解因数

(正整数x，分解成x=a1×……×an的形式，1<ai<=n，求分解种类数)

```python
def decompositions(n,minfactor):
    if n==1:
        return 1
    count=0
    for i in range(minfactor,n+1):
        if n%i==0:
        	count+=decomposition(n//i,i)
    return count
print(decomposition(x,2))
```

5.3.康托展开

对于一个排列P=[a1,a2,...,an]，其康托展开值x的计算公式为：

x=a1(n-1)!+a2(n-2)!+......+an-11!+an0!

其中ai是该位数字在剩余未使用数字中的排名（从0开始）

5.4.Catalan数

C0=1; C1=1; Cn=((4n-2)*Cn-1)//(n+1) 

y应用场景：有多少个合法出栈序列，合法括号序列数量，满节点二叉树数量，凸多边形的三角划分种类数，

n个运算符的表达式加括号方式数，n*n网格只能向上或向右的不穿过对角线的路径数量

5.5数的拆分问题

​	5.5.1.全部互异分拆/奇数分拆

```python
def divide_dif(n):
    dp=[[0]*(n+1) for _ in range(n+1)]
    for j in range(n+1):
        dp[0][j]=1
    for i in range(1,n+1):
        if i<j:
            dp[i][j]=dp[i][i]
        else:
            dp[i][j]=dp[i][j-1]+dp[i-j][j-1]#不含j和含一个j
    return dp[n][n]
```

​	5.5.2.全部分拆

```python
def divide(n):
    dp=[[0]*(n+1) for _ in range(n+1)]
    for j in range(n+1):
        dp[0][j]=1
    for i in range(1,n+1):
        if i<j:
            dp[i][j]=dp[i][i]
        else:
            dp[i][j]=dp[i][j-1]+dp[i-j][j]#不含j和含一个或多个j
    return dp[n][n]
```

​	5.5.3.划分为k个正整数的分拆

```python
def divide_k(n):
    dp=[[0]*(k+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][1]=1
    for i in range(1,n+1):
        for j in range(1,k+1):
        	if i>=j:
                dp[i][j]=dp[i-j][j]+dp[i-1][j-1]#不包含1和包含1
    return dp[n][k]
```

​	5.5.4.划分为k个非负整数（含0）的分拆

```python
#双dp，一个是划分为k个正整数，一个记录新的值
def divide0(n,k):
    dp1=[[0]*(k+1) for _ in range(n+1)]
    dp2=[[0]*(k+1) for _ in range(n+1)]
    for i in range(n+1):
        dp1[i][1]=1
        dp2[i][1]=1
    for i in range(1,n+1):
        for j in range(1,k+1):
            if i>=j:
                dp1[i][j]=dp1[i-j][j]+dp1[i-1][j-1]
            dp2[i][j]=dp2[i][j-1]+dp1[i][j]#包含0和不包含0
    return dp2[n][k]
```

**6.内置函数及用法**

**print**(value,sep='',end='\n') #参数均可修改

new_str=ori_str.**replace**(old,new,count) #默认替换全部

**bisect**  bisect_left(arr,num) #第一个>=目标值的位置 bisect_right(arr,num) #第一个>目标值的位置

if full_str.**startwith**(prefix): #判断前缀

**ord()** #得到字符的ASCⅡ码 **chr()** #ASCⅡ码转字符

**\\** 转义  print('\\\\') #输出\  

d[key]=d.**get**(key,default)+....

**itertools**  #以元组形式输出，需要实例化

product(,repeat=n) #元素可重复的排列

permutations(,n) #无重复元素的排列

combinations(,n) #有序无重复元素的组合

combinations_with_replace(,n) #有重复元素的组合

十进制数x的转换  bin(x)  #二进制  bin(x)[2:].zfill(n)  #去头定长的二进制  oct(x) #八进制   hex(x)  #十六进制

**math**

comb(n,k)	#不重复且无顺序地从n项中选择k项的方式总数

factorial(n)	#n的阶乘

gcd(*integers)	#整数参数的最大公约数

isqrt(n)	#非负整数n的整数平方根

lcm(*integers)	#整数参数的最小公约数

perm(n,k)	#无重复且有顺序地从n项中选择k项地方式总数

ceil(x)	#x向上取整

cbrt(x)	#x的立方根

exp(x)	#e的x次幂

exp2(x)	#2的x次幂

log(x,base)	#x的指定底数（默认为e）的对数

log2(x)	#x的以2为底的对数

log10(x)	#x的以10为底的对数

pow(x,y)	#x的y次幂(x，y均可以是小数)  #而自带的pow(x,y,mod)，x，y，mod必须是整数 

sqrt(x)	#x的平方根

dist(p,q)	#以坐标的可迭代对象形式给出的p和q两点之间的欧几里得距离  print(math.dist((1,1),(2,2))) #输出1.4142135623730951





