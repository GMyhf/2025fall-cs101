# **计概B CHEAT-SHEET**

24本经院 冯浩杰

## == 1. 杂项 ==
**zip()** 是 Python 的一个内置函数，用来把多个可迭代对象（如列表、元组、字符串等）“打包”在一起，生成由元组组成的迭代器。
```python
# from itertools import zip_longest
a = [1, 2, 3]
b = ['x', 'y']
print(list(zip_longest(a, b, fillvalue='-')))
# [(1, 'x'), (2, 'y'), (3, '-')]

transposed = list(map(list, zip(*matrix)))
```
#### 矩阵
```python
# 转置
transposed = list(map(list, zip(*matrix)))
# 矩阵乘法
multiply=[[sum([matrix[0][i][t]*matrix[1][t][j] for t in range(len(matrix[1]))]) for j in range(len(matrix[1][0]))] for i in range(len(matrix[0]))]
```
##### **最小堆**
```python
import heapq
heapq.heapify(my_list) #初始化
heapq.heappush(passed,-patrol[i][1])#pass是列表
fuel=-heapq.heappop(passed)
```
##### **欧拉筛**
```python
is_prime = [True] * (n + 1)
is_prime[0] = is_prime[1] = False
primes = []
for i in range(2, n + 1):
	if is_prime[i]:
		primes.append(i)
	for p in primes:
		if i * p > n:
			break
		is_prime[i * p] = False
		if i % p == 0:
			break
```
##### deque
appendleft 与 popleft    dq=deque([maxlen=])
|rotate(n)|将队列向右旋转 n 步（负值向左）|
|reverse()|原地反转队列|
|maxlen      |可选参数，设置队列最大长度|
selection = list(itertools.product([0, 1], repeat=6))
#### 排列组合与笛卡尔积
![[Pasted image 20251109000252.png]]
for i in range(times):  
    row = [(j + i) % n for j in range(n)]（==循环移位代码==）
#### 二分查找与bisect
bisect_left(a,x) 第一个>=  biscet_right 第一个严格大于索引
insort_left(a,x) x插入a中，位置在现有重复元素左边
```python
while left<=right:           #left<=right（等效做法）
	mid=(left+right)//2
	if mid<ans:（check(mid))：
		ans=mid					#  ans=mid
		left=mid+1           #  right=mid-1
	else:                   #else:left=mid+1
		right=mid-1
return left  #最小化最大值也是返回左边的（这种方法下）
# 最大化最小值
```

```python
# 最长严格递增序列
import bisect  
n = int(input())
lis, = map(int, input().split())
dp = [1e9]*n
for i in lis:#因为使用 `bisect_left`，<遇到相等元素会替换，不会延长序列）。如果要非严格递增（允许相等），应使用 `bisect_right`。
    dp[bisect.bisect_left(dp, i)] = i
print(bisect.bisect_left(dp, 1e8))
```
找最长上升子序列的长度，用left
>找最长下降子序列，先reverse，再用left
>如果是不降，用right
>如果是不升，先reverse，再用right
>看题目要求的最终结果是否需要相同元素的考虑，需要考虑用left，不需要用right
注意：left,right初始值错误可能导致结果错误！
卡特兰数：
h(n)=h(0)h(n−1)+h(1)h(n−2)+⋯+h(n−1)h(0)，其中 h(0)=1,h(1)=1（适用于 n≥2）。
例如：h(2)=h(0)h(1)+h(1)h(0)=1×1+1×1=2
```python
n=int(input())
dp=[0]*(n+1)
dp[1]=dp[0]=1
dp[2]=2
for i in range(3,n+1):
    for j in range(i):
        dp[i]+=dp[j]*dp[i-j-1]
print(dp[n])
```
### ==栈==
```python
#最长有效括号
stack=[-1] #设置‘哨兵值’
res=0
n=input()
for i,st in enumerate(n):
    if st=='(':
        stack.append(i)
    else:
        stack.pop()
        if not stack:
            stack.append(i) #右括号无法匹配，重新作为栈顶
        else:
            res=max(res,i-stack[-1])
print(res)
-----------
def longest_valid_brackets(s):
    pairs = {')': '(', ']': '[', '}': '{'}
    opening = set(pairs.values())
    stack = []
    valid = [False] * len(s)

    for i, ch in enumerate(s):
        if ch in opening:
            stack.append((ch, i))
        elif ch in pairs:
            if stack and stack[-1][0] == pairs[ch]:
                _, idx = stack.pop()
                valid[i] = True
                valid[idx] = True
            else:
                stack.clear()  # reset on mismatch

    # Now find longest consecutive True segment
    max_len = curr = 0
    for v in valid:
        if v:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len

# 示例
s = input()
print(longest_valid_brackets(s))
```
单调栈
```python
#柱状图中最大矩形
heights.append(-1)#哨兵，确保能全部弹出
stack=[]
res=[]
for i in range(len(heights)):
	if not stack:
		stack.append(i)
	else:
		k=stack[-1]
		if heights[k]<heights[i]:
			stack.append(i)
		else:
			while stack and heights[stack[-1]]>heights[i]:
				if len(stack)>=2:
					p=stack.pop()
					res.append((i-stack[-1]-1)*heights[p])
				else:
					p=stack.pop()
					res.append((i)*heights[p])
			stack.append(i)
return max(res)
```
今日化学论文
```python
stack = []  # 栈中存储元组：(当前累积的字符串, 当前的倍数)
current_str = ""
current_num = 0

for char in s:
	if char.isdigit():
		# 处理多位数情况，例如 100
		current_num = current_num * 10 + int(char)
	elif char == '[':
		# 遇到左括号，将现场“保护”起来，压入栈中
		stack.append((current_str, current_num))
		# 重置，准备记录括号内部的内容
		current_str = ""
		current_num = 0
	elif char == ']':
		# 遇到右括号，弹出最近的一个现场
		last_str, num = stack.pop()
		# 核心逻辑：之前的字符串 + (当前括号内的字符串 * 倍数)
		current_str = last_str + (current_str * num)
	else:
		# 普通小写字母
		current_str += char

print(current_str)
```

#### 元组
不可修改，只能拼接；可以索引；不能删除 
cmp(tuple1, tuple2)  比较两个元组元素。
#### 字典
dict.fromkeys(seq[, val])
创建一个新字典，以序列 seq 中元素做字典的键，val 为字典所有键对应的初始值
dict.has_key(key)  判断key是否在字典里
```python
from collections import defaultdict
```
#### 集合
1、set中的每一个元素必须是具有唯一性的hashable对象
3、set中的元素为完全无序，元素遍历时，与set插入元素的顺序完全无关……
5、set因为没有哈希值，所以不能被用作字典的键或其他集合的元素
添加元素： s.add(x) 如果元素已经存在，不进行任何操作；
		  s.update(x) 参数可以是列表，元组，字典等，X可以有多个
移除元素  s.remove(x) 如果元素不存在，会发生错误
		 s.discard(x) 元素不存在不会发生错误
		 s.pop（）随机删除集合中的一个元素
difference() 返回多个集合的差集 a.diiference(b) a-b
isdisjoint() 判断两个集合是否包含相同元素；返回布尔值
a.intersection(b)交集a&b a.union(b)并集 a|b 
a.symmetric_difference(b) 对称差集只属于a/b  a^ b
#### 进制
b:二进制 o:8 d:10 x:16小写 X：大写16 #+进制符号：带前缀
num=int(input())
print(format(num,'o'))  bin
使用 eval() 计算字符串表达式
#### 麦森数：
```python
# 1. 输出位数 
print(int(p * math.log10(2)) + 1)
# pow(a, b, c) 等价于 (a**b) % c，但效率极高
 res = pow(2, p, 10**500) - 1
 s = "{:0>500}".format(res) # 补齐500位
```
#### 求逆序数
```python
def count(arr):
    if len(arr) <= 1:
        return arr, 0
    mid = len(arr) // 2
    # 分治处理
    left_half, left_count = count(arr[:mid])
    right_half, right_count = count(arr[mid:])
    # 合并并统计
    merged, merge_count = merge(left_half, right_half)
    return merged,left_count+right_count+merge_count
def merge(left, right):
    result = []
    i = j = 0
    count = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
# 发现逆序：左侧当前元素及之后的所有元素都比 right[j] 大
            result.append(right[j])
            count += (len(left) - i)
            j += 1
    # 将剩余部分添加进去
    result.extend(left[i:])
    result.extend(right[j:])
    return result, count
```
## ==2. 递归==
#### 优化方法
1. 记忆化搜索
 ```python
from functools import lru_cache 
@lru_cache(None)
```
2. 增加递归深度限制
 ```python
import sys
sys.setrecursionlimit(1 << 30) # 将递归深度限制设置为 2^30
```
#### 递归中的回溯
回溯的基本结构：
```python
def backtrack(状态):
    if 满足结束条件:
        记录结果
        return
    for 选择 in 可选项:
        做出选择
        backtrack(新的状态)
        撤销选择  ← 这一步就是“回溯”
错误做法：（省略具体语法，只有选择部分）
def backtracking(con):
	backtracking(con+[k])  
	con.pop()#这是没有必要且有害的，因为con是函数的一个参数，上面的回溯是进行了一个新函数，这个函数的con没有变化；把pop()去掉就对了
正确写法（两种）：
	backtracking(con+[k]) #不加pop()
	
	con.append(k)
    backtracking(con)
    con.pop()
```

在递归中列表是可变的变量，如果使用回溯最好把列表当作递归中的一个参数调用
**回溯中去重的方法：约束顺序，引入start变量
```python
def dfs(n, k, start, path):  
    if k == 0:  
        if n == 0:  
            res.add(tuple(path))  
        return  
    for i in range(start, n + 1):  #只在start值之后进行循环
        dfs(n - i, k - 1, i, path + [i])
```

**全排列**
```python
有重复字符的情况需要区分，可以由位置索引进行排列
def backtracking(path):
    if len(path) == n:
        res.append(''.join(path))
        return
    for i in range(n):
        if used[i]:
            continue
        # 可选：跳过重复字符的重复分支以去重
        # if i>0 and st[i]==st[i-1] and not used[i-1]:
        #     continue
        used[i] = True
        backtracking(path + [st[i]])
        used[i] = False
        
def nextPermutation(nums: List[int]) -> None:
    i = len(nums) - 2
    while i >= 0 and nums[i] >= nums[i + 1]:
        i -= 1
    if i >= 0:
        j = len(nums) - 1
        while j >= 0 and nums[i] >= nums[j]:
            j -= 1
        
        nums[i], nums[j] = nums[j], nums[i]
    
    left, right = i + 1, len(nums) - 1
    while left < right:
        nums[left], nums[right] = nums[right], nums[left]
        left += 1
        right -= 1
```
**子集**
```python
可选写法：把sol当成一个函数的参数path
def backtracking(i, path):#原地修改path+回溯
            if i == n:
                res.append(path[:])     # 复制当前路径
                return
            path.append(nums[i])
            backtracking(i + 1, path) 
            path.pop()                 # 回溯
            # 不选 nums[i]
            backtracking(i + 1, path)
            #可以直接传入新列表，不需要pop，这样做不需要pop
            backtracking(i + 1, path + [nums[i]])#（注意这里一定要把num[i]变成列表再相加，不然会错误；也不要用append）
            backtracking(i + 1, path)
            
```
**分割回文子串**
```python
from functools import lru_cache
def partition(self, s: str) -> List[List[str]]:
	n = len(s)
	ans = []
	@lru_cache(None)  #记忆化
	def is_pal(i, j):
#判断回文子串，如果i>=j说明已经遍历完成；否则先判断两端，然后递归调用进行判断
		return i >= j or (s[i] == s[j] and is_pal(i + 1, j - 1))
def dfs(start, path):         #path是一个列表，作为内生变量防止更改
	if start == n:            #终止条件
		ans.append(path[:])
		return
for end in range(start, n): 
#对于每一个位置，先判断前面是不是回文子串，如果是，就递归调用判断后面的部分
	if is_pal(start, end):    #只有是回文子串才进行接下来的操作
		path.append(s[start:end + 1])
		dfs(end + 1, path)
		path.pop()
	dfs(0, [])
	return ans
```

## ==3. 图论==
#### bfs
```python
bfs=deque()  
inq={(0,0)}
bfs.append((0,0,0))
while bfs:  
    x,y,step=bfs.pop()   
    if maps[x][y]==1:  
        found=True  
        res=step  
        break  
    for dx,dy in dirs:  
        if 0<=x+dx<m and 0<=y+dy<n and maps[x+dx][y+dy]!=2:  
            if (x+dx,y+dy) not in inq:  
                bfs.appendleft((x+dx,y+dy,step+1))
                inq.add((x+dx,y+dy))
                
#dp替换inquiries：节约内存
dist = [[[-1] * 2 for _ in range(n)] for _ in range(n)]
queue = deque([(0, 0, 0)]) dist[0][0][0] = 0
# 5. 核心：遍历所有可能的下一步并“入队前去重” 
for nr, nc, ns in next_states:
 if dist[nr][nc][ns] == -1: # 如果这个状态没见过 
  dist[nr][nc][ns] = curr_steps + 1 # 立即打标（占位）
  queue.append((nr, nc, ns))
```

#### dijkstra 算法（带权图最短路径，要求没有负权）
```python
dist=[[float('inf')]*n for _ in range(m)]
dist[x1][y1]=0  
ways=[(0,(x1,y1))]
while ways:  
    dis,pot=heapq.heappop(ways)  
    x,y=pot  
    if dis>dist[x][y]:  
        continue  
    for dx,dy in dirs:  
        if 0<=dx+pot[0]<m and 0<=dy+pot[1]<n:  
            if maps[x+dx][y+dy]=='#':  
                continue
            if dis+cost<dist[x+dx][y+dy]:  
			    dist[x+dx][y+dy]=dis+cost  
		       heapq.heappush(ways, (dis + cost,(x+dx,y+dy)))
```
#### 并查集与带权并查集
1. find
```python
case=1  
self=[]  
set_count=0  
rank=[]  
def find(i):  
    global self  
    if self[i]!=i:
	    origin_fa = parent[x]  #记录父节点
        self[i]=find(self[i])
        # 核心：当前权值加上父亲到根节点的权值
        value[x] += value[origin_fa]  
    return self[i]
```
2. union
```python
def union(self, i, j):
      def union(i,j):  
    global self,rank,set_count  
    root_i=find(i)  
    root_j=find(j)  
    if root_i!=root_j:  
        if rank[root_i]>rank[root_j]:  
            self[root_j]=root_i  
        elif rank[root_i]<rank[root_j]:  
            self[root_i]=root_j  
        else:  
            self[root_i]=root_j  
            rank[root_i]+=1  
        set_count-=1
        # 根据推导公式更新根节点的权值
        value[rootX] = value[y] + w - value[x]
for i in range(1, n + 1):  #注意路径压缩不完全问题
    if tree[i - 1] == i:   #重新计数
        ans.append(i)
```
- **初始化**：记得 `parent[i] = i` 且 `value[i] = 0`（或者乘法逻辑下初始化为 1）。
- **先 find 再访问**：在判断两个节点的关系前，**必须先对两个节点分别执行一次 `find`**，确保它们的 `value` 都已经更新为相对于根节点的值。
- **模运算**：如果是处理循环关系（如食物链），所有的加法后面都要加上 `% mod`，为了防止 Python 出现负数，建议写成 `(a - b + mod) % mod
节点 $i$ 到根的距离 = (节点 $i$ 到原父亲的距离) + (原父亲到根的距离)
## ==贪心==
#### 区间问题
1. 区间合并
【**步骤⼀**】：按照区间**左端点**从⼩到⼤排序。
【**步骤⼆**】：维护前⾯区间中最右边的端点为ed。从前往后枚举每⼀个区间，判断是否应该将当前区间视为新区间。对区间【l,r】
如果l<=ed,说明有交集，不需要增加区间个数，但ed=max(ed,r)
2. 选择不相交区间
【步骤⼀】：按照区间**右端点**从⼩到⼤排序。
【步骤⼆】：从前往后依次枚举每个区间。
l<=ed,说明有交集，直接跳过；l>ed,选中并且设置ed=r
3. 区间覆盖问题
【步骤⼀】：按照区间左端点从⼩到⼤排序。
【步骤⼆】：从前往后依次枚举每个区间，在所有能覆盖当前⽬标区间起始位置start的区间之中，选择右端点最⼤的区间。
最后将目标区间的start改为r
```python
#oj-世界杯只因-27104
def min_cameras(ranges):
    n=len(ranges)
    mx=max(ranges)
    curr=0
    num=0
    while curr<n:
        next=curr+ranges[curr]+1
        for i in range(max(0,curr-mx),min(n,curr+mx+1)):
            next=max(next,i+ranges[i]+1)
        num+=1
        curr=next
    return num
```
1. 区间分组问题
【步骤⼀】：按照区间左端点从⼩到⼤排序。
【步骤⼆】：从前往后依次枚举每个区间，判断当前区间能否被放到某个现有组⾥⾯。
（即判断是否存在某个组的右端点在当前区间之中。如果可以，则不能放到这⼀组）
如果所有 m 个组⾥⾯没有组可以接收当前区间，则当前区间新开⼀个组，并把⾃⼰放进去；如果存在可以接收当前区间的组 k，则将当前区间放进去，并更新当前组边界
```python
def min_host(n,ranges):
    events=[]
    for i in range(n):
        events.append((ranges[i][0],1)) #新添1，表示出现一个起点时主持人数加1
        events.append((ranges[i][1],-1)) #新添-1，表示出现一个终点时主持人数减1
        #最后统计主持人数聚集最多的个数，就是答案
    events.sort(key=lambda x:(x[0],x[1]))
    min_hosts=0
    curr=0
    
    for time,num in events:
        curr+=num
        min_hosts=min(min_hosts,curr)
    return min_hosts
```
## dp
#### 整数划分问题
1. n划分为k份，允许有0，不考虑顺序
```python
f(n,k)=f(n,k-1)+f(n-k,k) #是否有空盘子分类；不允许空去左边
边界：
dp[0][j]=1 dp[i][1]=1 dp[i][j]=dp[i][i](i<j)
```
2. 把n用不超过k的整数划分
把问题看作用面值为1-k的硬币凑出n
```python
for i in range(1, k + 1): #遍历面值
	for j in range(i, n + 1): # 遍历背包容量 j
		dp[j] += dp[j - i]
return dp[n]
边界：dp[0]=1
```
3. 把n划分为若干个正整数，**考虑顺序**
4：4=3+1=1+3=2+2=2+1+1=1+2+1=1+1+2=1+1+1+1 共8种
```python
def divide2(n):
    dp=[1]+[0]*n
    for i in range(1,n+1):         #每个容量（每个n）
        for j in range(1,i+1):   #每个可能划分出的数字
            dp[i]+=dp[i-j]
    return dp[-1]
```
4. 把n划分为若干个不同的正整数，不考虑顺序 --> 01背包
4：4=3+1 共1种
```python
def divide3(n):
    dp=[1]+[0]*n
    for i in range(1,n+1):
        for j in range(n,i-1,-1):
            dp[j]+=dp[j-i]
    return dp[-1]
```
#### 背包问题
1. 0-1背包问题（双限制）
```python
dp = [[0] * (m + 1) for _ in range(n + 1)]
for num,hurts in animals:
    for i in range(n, num-1, -1):#逆序遍历，边界-1
        for j in range(m, hurts-1, -1):
            dp[i][j]=max(dp[i][j],dp[i-num][j-hurts]+1)
max_count=dp[n][m]
#计算最大收获时某一限制消耗的最小值
for j in range(m+1):
    if dp[n][j]==max_count:
        min_hurt=j
        break
```
2. 完全背包问题
```python
for i in range(n):  # 遍历每一个物品
	v = weights[i]    w = values[i]
	# 注意：这里是从 v 到 V 的正序遍历
	for j in range(v, V + 1):
		dp[j] = max(dp[j], dp[j - v] + w)
```
3. 多重背包问题（每种物品数量有限）
```python
def multiple_knapsack_binary(V, weights, values, nums):
    dp = [0] * (V + 1)
    # 构造拆分后的新物品清单
    new_weights = []
    new_values = []
    for i in range(len(weights)):
        v, w, s = weights[i], values[i], nums[i]
        # 二进制拆分过程
        k = 1
        while k <= s:
            new_weights.append(k * v)
            new_values.append(k * w)
            s -= k
            k *= 2  # 1, 2, 4, 8...
        # 放入剩余的部分
        if s > 0:
            new_weights.append(s * v)
            new_values.append(s * w)

    # 对拆分后的所有物品跑一遍 0-1 背包
    for i in range(len(new_weights)):
        .........
            
    return dp[V]
```
4. 最小消耗问题
```python
# 1. 初始化：求最小则初始化为无穷大
# dp[i][j] 表示至少获得 i 单位资源1和 j 单位资源2的最小代价
dp = [[float('inf')] * (M + 1) for _ in range(V + 1)]
dp[0][0] = 0 
for v, m, w in items:
	# 2. 逆序遍历（0-1背包思想）
	for i in range(V, -1, -1):
		for j in range(M, -1, -1):
			# 3. 核心技巧：下标不能小于0
			# 如果当前物品提供的量 v 大于需求 i，则转移到 dp[0] 状态
			# 因为“至少得到 -2 的量”等同于“至少得到 0 的量”
			next_i = max(0, i - v)
			next_j = max(0, j - m)
			if dp[next_i][next_j] != float('inf'):
			dp[i][j] = min(dp[i][j], dp[next_i][next_j] + w)
```
#### 数位dp
```python
s = str(n)
@lru_cache(None)
def dfs(pos, prev, is_limit, is_num):
# pos：记录当前位数 prev:根据题目要求计数 is_num：是否只有前导0
	if pos == len(s):
		return 1 if is_num else 0 # 填写完毕，如果填过数字则算一个合法方案
	res = 0
	# 确定当前位能填的最大数字；如果前一位数字最大，则islimit真，受限
	up = int(s[pos]) if is_limit else 9
	for d in range(up + 1):
		# 判断逻辑：根据题目要求修改
		if d == 4: continue # 比如不要数字 4
		res += dfs(
			pos + 1, 
			d, 
			is_limit and (d == up), 
			is_num or (d > 0)
		)
	return res
return dfs(0, -1, True, False)
```
#### 带连续限制的线性计数 DP
```python
from functools import lru_cache  
n,m=map(int,input().split())  
@lru_cache(maxsize=None)  
def dfs(idx,k):  
    if k>=m:        
	    return 0    
	if idx==n:        
		return 1    
	res=dfs(idx+1,0)    
	res+=dfs(idx+1,k+1)    
	return res
print(dfs(0,0))  
#递归方法，更容易理解，对每一位分类讨论放或者不放
dp=[1]*(n+1)  
for i in range(1,n+1):  
    if i<m:  
        dp[i]=2*dp[i-1]  #没有到达限制，随便放
    elif i==m:  
        dp[i]=2*dp[i-1]-1  #减去全放
    else:#减去刚好非法的情况，即i-m不放，（不然在i-1非法）i-m+1到i全放
        dp[i]=2*dp[i-1]-dp[i-m-1]  
print(dp[n])
```
#### 打家劫舍模型（不能连续选择）
```python
# 初始状态
prev_no, prev_yes = 0, 0
for v in values:
    # 当前不选 = 上一个选或不选的最大值
    curr_no = max(prev_no, prev_yes)
    # 当前选 = 上一个必须没选 + 当前价值
    curr_yes = prev_no + v
    # 更新
    prev_no, prev_yes = curr_no, curr_yes
return max(prev_no, prev_yes)
```
####  kadane算法 
线性计算一维数组最大连续和；可以计算最大子矩阵
```python
def kadane(factors):  
    close=far=factors[0]  
    for x in factors[1:]:  
        close=max(x,x+close)  #遍历，如果能更大就加入，否则重新开始
        far=max(far,close)    #不断维护最大值
    return far
res=[]  
for i in range(n):  
    for j in range(n-i):  
        lines=[sum(matrix[k][j:j+i+1]) for k in range(n)]  #二维kadane算法，把若干列合并
        res.append(kadane(lines))  
print(max(res))
```
#### 区间dp
```python
# n 为序列长度
dp = [[0] * n for _ in range(n)]

# 1. 枚举区间长度 len (从 2 开始，长度为 1 的通常是初始值)
for length in range(2, n + 1):
    # 2. 枚举左端点 i
    for i in range(n - length + 1):
        j = i + length - 1  # 自动计算右端点 j
        
        # 3. 枚举分割点 k (i <= k < j)
        # 初始化 dp[i][j] 为无穷大或 0
        dp[i][j] = float('inf') 
        for k in range(i, j):
            # 转移方程：小区间的和 + 合并这次区间的代价
            cost = dp[i][k] + dp[k+1][j] + calculate_cost(i, j)
            dp[i][j] = min(dp[i][j], cost)
```
---
### IV.SORTING
（参考汤伟杰大佬的chaet-sheet）
#### 1.冒泡排序
```python
def bubble_sort(s):
    n=len(s)
    f=True
    for i in range(n-1):
        f=False
        for j in range(n-i-1):
            if s[j]>s[j+1]:
                s[j],s[j+1]=s[j+1],s[j]
                f=True
        if f==False:
            break
    return s
```
#### 2.归并排序 --> 递归
```python
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
```python
#lc-LCR085-括号生成
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        ans=[]
        path=['']*(n*2) #对path原地修改，然后放入ans；否则要考虑浅拷贝的问题
        def dfs(i,left): #i:填充的索引; left:左括号的个数
            if i==n*2: 
                ans.append(''.join(path))
            if left<n: #如果左括号的个数小于n个(可以填一个左括号)
                path[i]='('
                dfs(i+1,left+1)
            if i-left<left: #如果右括号的个数小于左括号的(可以填一个右括号)
                path[i]=')'
                dfs(i+1,left)
        dfs(0,0)
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
#### 4. 冒泡排序
```python
for i in range(len(s)-1):
    for j in range(len(s)-i-1):
        if str(s[j])+str(s[j+1])<str(s[j+1])+str(s[j]):
            s[j],s[j+1]=s[j+1],s[j]
            
#from functools import cmp_to_key

# 1. 定义你的两两比较逻辑
def my_compare(a, b):
    # a 和 b 是列表中的两个元素（字符串）
    if a + b > b + a:
        return -1  # 返回负数表示 a 应该排在前面
    elif a + b < b + a:
        return 1   # 返回正数表示 a 应该排在后面
    else:
        return 0   # 相等

# 2. 使用 cmp_to_key 转换并排序
sort=sorted(nums, key=cmp_to_key(my_compare))
#counter函数进行次数统计
from collections import Counter

# 统计列表
c = Counter(['a', 'b', 'c', 'a', 'b', 'a'])
# 结果: Counter({'a': 3, 'b': 2, 'c': 1})

# 统计字符串
s = Counter("banana")
# 结果: Counter({'a': 3, 'n': 2, 'b': 1})
```
#### 5. 拓扑排序
```python
from collections import deque

def topological_sort(n, edges):
    in_degree = [0] * n  # 记录每个点的入度
    adj = [[] for _ in range(n)]  # 邻接表记录边
    
    for u, v in edges:
        adj[u].append(v)
        in_degree[v] += 1
    
    # 1. 初始入度为0的点入队
    queue = deque([i for i in range(n) if in_degree[i] == 0])
    result = []
    
    while queue:
        u = queue.popleft()
        result.append(u)
        # 2. 遍历邻居，模拟断开连线
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:  # 3. 入度变0的点入队
                queue.append(v)
                
    return result if len(result) == n else "图中存在环"
```
#### 2.前缀和的特殊用法(哈希表)
使用prefix和prefix_map来记录已有的前缀和，从而判断子串和为0的子串个数；或找相同前缀和数字出现的最远位置
例题：
```python
#找出不重叠的和为0的子序列个数，一旦找到就将prefixed集合清空
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

自己犯过的易错点：
1. 在 Python 中，字符串的比较是从左到右逐个字符比较其 ASCII 码（即**字典序**），不考虑位数；不能直接max(dp)
2. OJ的pylint是静态检查，有时候报的compile error不对。解决方法有两种，如下：(1）第一行加# pylint: skip-file(2）方法二：如果函数内使用全局变量（变量类型是immutable，如int），则需要在程序最开始声明一下。如果是全局变量是list类型，则不受影响。
3. 警惕边界情况（Corner Cases）
在提交前，快速脑补一下这些特殊输入：
- $N = 1$
- 时程序会崩吗？
- 如果没有符合条件的计划（如寒假生活里全是 2.20 以后的），输出是 0 还是报错？
- 坐标系问题：$x, y$ 的范围是否是从 $0$ 到 $n-1$？
 4. **不要在死胡同里打转**：如果一个 Bug 找了 15 分钟没头绪，立刻跳过去看下一题。有时候换个思路，或者回头再看，灵感就来了。
- **理解题目意图**：看到“中位数”想堆，看到“最短距离”想 BFS，看到“覆盖/重叠/最大价值”想 DP，看到“先后依赖”想拓扑排序。
- **一定要认真读题！** WA时先重读一遍题目
1. **不到最后一刻，千万不要放弃！**
