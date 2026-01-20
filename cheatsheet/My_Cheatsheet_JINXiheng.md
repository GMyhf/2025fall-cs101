## **==认真读题==**
## **==留意变量范围，考虑算法的时间复杂度是否够用==**

## Pycharm 调试快捷键

单步跳过：F8
步入函数：F7
跳出当前函数：Shift + F8
运行到光标处：Alt + F9
继续执行：F9

# 0 少量语法

## 0.1 输入输出

不定行输入的几种读法：

```python
#try...except捕获异常
while True:
	try:
		#正常读取
	except EOFError:
		break
		
#利用sys逐行读取
import sys
for line in sys.stdin:
	line = line.strip()
	#处理每一行

#同样适用于一般的输入读取特别是大量数据输入流优化
data = sys.stdin.read()
lines = data.splitlines()
for line in lines:
	#处理每一行
```

==注意：==当然，最后一种方法中也可以使用 $data.split()$ 后用 $data[idx]$ 来得到输入中的各个项。

## 0.2 可变与不可变对象

常用不可变对象：整数，浮点数，字符串，元组 等。

常用可变对象：列表，字典，集合。

==重要：==可变对象作为参数传入函数，若在函数内被修改，会影响外部变量。


# 1 若干值得一抄的常用技术/方法
## 1.1 二分查找：一类二分查找问题的通法

不失一般性，我们假设：$key(arr[idx])$ 关于 $idx$ 单调增加。否则，若单调减少，取负的函数值即可。

### 1.1.1 寻找arr中从左往右第一个使得key函数值>x（或>=x）的元素

直接参考bisect库源码即可。注意“循环不变量”的思想：要保证目标下标始终位于闭区间 $[lo, hi]$ 中，这样更新边界的时候就不会搞错“加不加1”“减不减1”这类问题。

```python
#寻找第一个使key值>x的元素，返回a中下标
#若无满足条件的下标，返回len(a)
def bisect_strict_first(a, key):
	lo, hi = 0, len(a)
	while lo < hi:
		mid = (lo + hi) // 2
		if key(a[mid]) > x:
			hi = mid
		else:
			lo = mid + 1
    return lo

#寻找第一个使key值>=x的元素，返回a中下标
#若无满足条件的下标，返回len(a)
def bisect_nonstrict_first(a, key):
	lo, hi = 0, len(a)
	while lo < hi:
		mid = (lo + hi) // 2
		if key(a[mid]) < x:
			lo = mid + 1
		else:
			hi = mid
	return lo
```

==注意：== 该源码实际上可以用于寻找合适的插入x的位置，即使数组中元素全部<=x（或全部<x），此代码仍然有效。初始化 $hi$ 为 $len(a)$ 保证了这一点。

稍加修改，我们就能得到另一类镜像问题的解法：

### 1.1.2 寻找arr中从左往右最后一个使得key函数值<x（或<=x）的元素

```python
#寻找最后一个使key值<x的元素，返回a中下标
#若无满足条件的下标，返回-1
def bisect_strict_last(a, key):
	lo, hi = -1, len(a) - 1
	while lo < hi:
		mid = (lo + hi + 1) // 2
		if key(a[mid]) < x:
			lo = mid
		else:
			hi = mid - 1
	return hi
	
#寻找最后一个使key值《=x的元素，返回a中下标
#若无满足条件的下标，返回-1
def bisect_nonstrict_last(a, key):
	lo, hi = -1, len(a) - 1
	while lo < hi:
		mid = (lo + hi + 1) // 2
		if key(a[mid]) > x:
			hi = mid - 1
		else:
			lo = mid
	return hi
```

## 1.2 并查集

基本用法可参考M02524: 宗教信仰 一题的代码：

```python
ans=[]  
  
while True:  
    n,m=map(int,input().split())  
    if not n:  
        break  
  
    stu=[i for i in range(n)]  
  
    def get_top(i):  
        if stu[i]!=i:  
            return get_top(stu[i])  
        else:  
            return i  
  
    for _ in range(m):  
        a,b=map(int,input().split())  
        stu[get_top(b-1)]=get_top(a-1)  
  
    for i in range(0,n):  
        stu[i]=get_top(i)  
  
    ans.append(len(set(stu)))  
  
for i in range(1,len(ans)+1):  
    print(f'Case {i}: {ans[i-1]}')
```


# 2 递归

貌似没问题的递归代码爆RE，考虑一下递归深度。
递归爆栈可考虑调整递归深度限制：

```python
import sys
sys.setrecursionlimit(1<<30)
```

避免重复计算子问题：

```python
from functools import lru_cache
@lru_cache(maxsize = None)
```
这个必须要在函数有返回值且不会造成其他变量改变时，才能使用。
## 2.1 回溯

回溯算法的一般步骤如下面伪代码所示。需要注意以下几点：

- 回溯函数返回值以及参数
- 回溯函数终止条件
- 回溯搜索的遍历过程（横向和纵向）

```python
def backtracking(parameters):
	if ... : #终止条件
		#存放结果
		return
	for ... : #选择本层集合中的元素
		#处理节点
		backtracking(...) #参数一般会包含做过的选择的信息
		#撤销处理结果
```

记得考虑能否通过剪枝（去掉不可能的选择）来提高效率。

例：sy136. 组合II 给定一个长度为n的序列，其中有n个互不相同的正整数，再给定一个正整数k，求从序列中任选k个的所有可能结果

```python
n,k=map(int,input().split())  
data=list(map(int,input().split()))  
check=[1]*n  
  
ans_array,now_array=[],[]  
def all_array():  
    judge=1  
    for i,num in enumerate(data):  
        if len(now_array)==k and judge:  
            ans_array.append(now_array[:])  
            judge=0  
        elif check[i] and (len(now_array)==0 or num>max(now_array)):  
            now_array.append(num)  
            check[i]=0  
            all_array()  
            now_array.pop()  
            check[i]=1  
  
all_array()  
  
for ans in ans_array:  
    print(*ans,sep=' ')
```

# 3. DP

注意以下步骤：

- 确定dp数组以及下标的含义
- 确定递推公式
- dp数组如何初始化
- 确定遍历顺序
- 举例推导dp数组

# 4. 简单图论

## 4.1 深度优先搜索

可以参考递归、回溯部分内容。

## 4.2 广度优先搜索

例：sy321迷宫最短路径

思路：每次尝试走一步，并用一个trace记录每一步的上一步来自哪里，直到走到右下角；然后用trace反向走回起点，就可以得到最优路径了。

```python
from collections import deque  
moves=deque([(-1,0),(1,0),(0,1),(0,-1)])  
  
n,m=map(int,input().split())  
area=deque([deque([1]*(m+2))])  
for _ in range(n):  
    area.append(deque([1]+list(map(int,input().split()))+[1]))  
area.append(deque([1]*(m+2)))  
  
trace=deque(deque([(-1,-1)]*(m+2)) for _ in range(n+2))  
trail=deque()  
trail.append((1,1))  
next_trail=set()  
judge=1  
  
while judge:  
    for now in trail:  
        for move in moves:  
            n_x=now[0]+move[0]  
            n_y=now[1]+move[1]  
            if not area[n_x][n_y]:  
                area[n_x][n_y]=1  
                next_trail.add((n_x,n_y))  
                trace[n_x][n_y]=(now[0],now[1])  
            if n_x==n and n_y==m:  
                judge=0  
                break  
    trail.clear()  
    trail.extend(next_trail)  
    next_trail.clear()  
  
path=deque([(n,m)])  
while True:  
    path.appendleft(trace[path[0][0]][path[0][1]])  
    if path[0]==(1,1):  
        break  
  
for step in path:  
    print(step[0],step[1])
```

当然，此题的visited用二维布尔数组亦可。

## 4.3 Dijkstra算法

贪心扩散算法，每一步都计算邻居成本并更新未定点，并选择所有未定点中成本最小的，一旦选定，就把下一个点确定为最小成本，该点成为已确定，如此遍历即可。

# CE处理
OJ的pylint是静态检查，有时候报的compile error不对。解决方法有两种，如下：
1）第一行加# pylint: skip-file
2）方法二：如果函数内使用全局变量（变量类型是immutable，如int），则需要在程序最开始声明一下。如果是全局变量是list类型，则不受影响。

# heapq

### 1. 基本操作
```python
import heapq

# 创建堆（使用列表）
heap = []

# 添加元素
heapq.heappush(heap, 5)      # [5]
heapq.heappush(heap, 2)      # [2, 5] - 自动调整
heapq.heappush(heap, 8)      # [2, 5, 8]

# 弹出最小元素
smallest = heapq.heappop(heap)  # 2, 堆变成[5, 8]
```

### 2. 列表转堆
```python
nums = [3, 1, 4, 1, 5]
heapq.heapify(nums)  # 原地转换: [1, 1, 4, 3, 5]
```

### 3. 组合操作
```python
heap = [3, 5, 8]

# 先推后弹（更高效）
result = heapq.heappushpop(heap, 4)  # 3, 堆变成[4, 5, 8]

# 先弹后推
result = heapq.heapreplace(heap, 3)  # 4, 堆变成[3, 5, 8]
```

### 1. 找最大/最小的 n 个元素
```python
nums = [1, 8, 2, 23, 7, -4, 18]

# 最大的3个
largest = heapq.nlargest(3, nums)  # [23, 18, 8]

# 最小的3个  
smallest = heapq.nsmallest(3, nums)  # [-4, 1, 2]

# 按自定义key
students = [{'name': 'A', 'score': 85}, {'name': 'B', 'score': 92}]
top = heapq.nlargest(1, students, key=lambda x: x['score'])
```

### 2. 合并有序序列
```python
list1 = [1, 3, 5]
list2 = [2, 4, 6]
merged = heapq.merge(list1, list2)  # 返回迭代器: 1,2,3,4,5,6
```

## 四、最大堆技巧
`heapq` 只有最小堆，实现最大堆需存储负值：
```python
max_heap = []
heapq.heappush(max_heap, -10)  # 存负值
heapq.heappush(max_heap, -5)
largest = -heapq.heappop(max_heap)  # 10 (取负还原)
```

## 六、特性总结
- **时间复杂度**：
  - `heappush/heappop`: O(log n)
  - `heapify`: O(n)
  - `nlargest/nsmallest`: O(n log k)
- **原地操作**：所有函数直接修改原列表
- **最小堆**：`heap[0]` 总是最小元素
- **使用普通列表**：无需特殊数据结构

# deque 

```python
from collections import deque
```

```python
# 空 deque
dq = deque()
dq = deque([1, 2, 3])      # 从列表创建
dq = deque("hello")        # 从字符串创建: ['h','e','l','l','o']
dq = deque(maxlen=3)       # 固定大小，满时自动丢弃另一端元素
```
### 1. 两端操作（O(1)时间）
```python
dq = deque([1, 2, 3])

# 右端操作
dq.append(4)        # [1, 2, 3, 4]
right = dq.pop()    # 返回4, [1, 2, 3]

# 左端操作（deque特有）
dq.appendleft(0)    # [0, 1, 2, 3]
left = dq.popleft() # 返回0, [1, 2, 3]
```

### 2. 批量扩展
```python
dq = deque([1, 2])

# 右端扩展
dq.extend([3, 4])   # [1, 2, 3, 4]

# 左端扩展（注意顺序会反转）
dq.extendleft([-1, 0])  # [0, -1, 1, 2, 3, 4]
```

### 1. 旋转 `rotate()`
```python
dq = deque([1, 2, 3, 4, 5])

dq.rotate(2)    # 向右旋转: [4, 5, 1, 2, 3]
dq.rotate(-1)   # 向左旋转: [5, 1, 2, 3, 4]
```

### 2. 固定大小 deque
```python
dq = deque(maxlen=3)
dq.append(1)  # [1]
dq.append(2)  # [1, 2]
dq.append(3)  # [1, 2, 3]
dq.append(4)  # [2, 3, 4]（自动丢弃1）
```

```python
dq = deque([1, 2, 3, 2, 4])

len(dq)            # 5
dq.count(2)        # 2（计数）
dq.index(3)        # 2（查找索引）
dq.remove(2)       # 删除第一个2: [1, 3, 2, 4]
dq.clear()         # 清空: []
```
- 频繁两端操作用 **deque**
- 频繁随机访问用 **list**  
- 固定大小需求用 `maxlen` 参数
- 线程安全：deque 原子操作适合多线程

# bisect 

### 1. `bisect_left(a, x, lo=0, hi=len(a))`
返回第一个 **≥ x** 的元素位置
```python
arr = [1, 3, 5, 7, 9]
pos = bisect.bisect_left(arr, 4)  # 2 (在5之前)
```

### 2. `bisect_right(a, x, lo=0, hi=len(a))` 或 `bisect()`
返回第一个 **> x** 的元素位置
```python
arr = [1, 3, 5, 7, 9]
pos = bisect.bisect_right(arr, 5)  # 3 (在5之后)
pos = bisect.bisect(arr, 5)        # 同上
```

### 1. `insort_left(a, x, lo=0, hi=len(a))`
插入元素并保持有序，相等时插入**左边**
```python
arr = [1, 3, 5, 7, 9]
bisect.insort_left(arr, 4)  # [1, 3, 4, 5, 7, 9]
```

### 2. `insort_right(a, x, lo=0, hi=len(a))` 或 `insort()`
插入元素并保持有序，相等时插入**右边**
```python
arr = [1, 3, 5, 7, 9]
bisect.insort_right(arr, 5)  # [1, 3, 5, 5, 7, 9]
```

| 函数 | 相等元素位置 | 返回值意义 |
|------|------------|-----------|
| `bisect_left()` | 插入左边 | 第一个 ≥ x 的位置 |
| `bisect_right()` | 插入右边 | 第一个 > x 的位置 |
| `insort_left()` | 插入左边 | 实际插入操作 |
| `insort_right()` | 插入右边 | 实际插入操作 |

## 五、关键特性
- **时间复杂度**：O(log n) 二分查找
- **前提条件**：列表必须**有序**
- **原地操作**：插入函数直接修改原列表
- **返回值**：查找函数返回位置索引，插入函数不返回值

```python
bisect_left(a, x, lo=0, hi=len(a))
# a: 有序列表
# x: 要查找/插入的值
# lo: 搜索起始索引（默认0）
# hi: 搜索结束索引（默认len(a)）
```


## lambda函数
```python
sort() #稳定的从小到大排序，如果列表存储的是多元元组，则依次按照每个元组的元素进行排序，且稳定
#如果想自行按照元组的元素顺序排序，可以使用lambda函数
s=[(1,2),(3,1),(4,5),(2,5)]
#按照第二个元素排序
s.sort(key=lambda x:x[1]) #[(3, 1), (1, 2), (2, 5)]
#按照第二个元素为首要升序排序，第一个元素为次要升序排序
s.sort(key=lambda x:(x[1],x[0])) #[(3, 1), (1, 2), (2, 5), (4, 5)]
#按照第二个元素为首要降序排序，第一个元素为次要升序排序
s.sort(key=lambda x:(-x[1],x[0])) #[(2, 5), (4, 5), (1, 2), (3, 1)]
#-----------------------------#
#如果想对数字按照字典序组合排序，得到最大最小整数，可以冒泡可以匿名
s=[9989,998]
#冒泡
for i in range(len(s)-1):
    for j in range(len(s)-i-1):
        if str(s[j])+str(s[j+1])<str(s[j+1])+str(s[j]):
            s[j],s[j+1]=s[j+1],s[j]
#lambda函数
s=sorted(s,key=lambda x: str(x)*10,reverse=True)
#-----------------------------#
#对字典的键值对进行排序，与列表存储元组差不多
d={3:34,2:23,9:33,10:33}
dd=dict(sorted(d.items(),key=lambda x:(x[1],-x[0]))) #{2: 23, 10: 33, 9: 33, 3: 34}
```


# zip

`zip()` - 将多个可迭代对象**打包**成元组
### 基本用法：
```python
names = ['Alice', 'Bob', 'Charlie']
ages = [25, 30, 35]

# 打包成 (name, age) 对
zipped = zip(names, ages)
print(list(zipped))  
# [('Alice', 25), ('Bob', 30), ('Charlie', 35)]
```
### 同时遍历多个列表：
```python
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")
```
### 创建字典：
```python
person_dict = dict(zip(names, ages))
# {'Alice': 25, 'Bob': 30, 'Charlie': 35}
```
### 长度不等时：
```python
# 以最短的为准
list(zip([1,2,3], ['a','b']))  # [(1,'a'), (2,'b')]
```
### 解压：
```python
zipped = [('a',1), ('b',2), ('c',3)]
letters, numbers = zip(*zipped)
print(letters)  # ('a', 'b', 'c')
print(numbers)  # (1, 2, 3)
```


# 欧拉筛

```python
def euler_sieve(n):
    is_prime = [True] * (n + 1)
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
    return primes
```