**1.greedy**
(1)bubble sort
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        # 标记是否发生了交换
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                # 交换元素
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        # 如果没有发生交换，说明数组已经排序完成
        if not swapped:
            break
    return arr
```
eg.最大最小整数
```python
n = int(input())  
nums = list(input().split())  
groups = {str(i): [] for i in range(1,10)}  
for num in nums:  
    if num and num[0].isdigit() and num[0] in groups:  
        groups[num[0]].append(num)  
def number(arr):  
    s = ''  
    for i in range(len(arr)):  
        s += arr[i]  
    return int(s)  
maximum = ''  
minimum = ''  
def bubble_sort(arr,ascending=False):  
    k = len(arr)  
    for i in range(k):  
        swapped = False  
        for j in range(0, k - i - 1):  
            a = number(arr)  
            arr[j], arr[j + 1] = arr[j + 1], arr[j]  
            b = number(arr)  
            if not ascending:  
                if a <= b:  
                    swapped = True  
                else:  
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]  
            else:  
                if a >= b:  
                    swapped = True  
                else:  
                    arr[j], arr[j + 1] = arr[j + 1], arr[j]  
        if not swapped:  
            break  
    return arr  
  
# 为了最大值，我们需要从 '9' 到 '1' 排序
sorted_groups_desc = [
    "".join(bubble_sort(groups[str(i)].copy(), False))
    for i in range(9, 0, -1)
]
maximum = "".join(sorted_groups_desc)

# 为了最小值，我们需要从 '1' 到 '9' 排序
sorted_groups_asc = [
    "".join(bubble_sort(groups[str(i)].copy(), True))
    for i in range(1, 10)
]
minimum = "".join(sorted_groups_asc)

print(maximum,minimum)
```
(2)two pointers：初始化-判断-移动-更新
I.从两端往中间收缩
eg.XXXXX：找靠近两端的x的倍数
```python
l = 0  
r = len(a)-1  
left = None  
right = None  
while l <= r and (left is None or right is None):  
    if left is None:  
        if a[l]%x != 0:  
            left = l  
        else:  
            l += 1  
    if right is None:  
        if a[r]%x != 0:  
            right = r  
        else:  
            r -= 1
```
II.从中间往两端扩张：sliding window
eg.一一相依：找0个数不超过k的最长子列
```python
def longest_ones(n, k, s):  
    l, r = 0, -1  # 初始化左右指针  
    zeroCount = 0  # 记录当前窗口内0的个数  
    max_length = 0  # 记录最大连续1的个数  
    while r < n:  
        if zeroCount <= k:  # 如果当前窗口内0的个数不超过k  
            max_length = max(max_length, r - l + 1)  # 更新最大长度  
            r += 1  # 扩展右边界  
            if r < n and s[r] == '0':  # 如果新加入的字符是0  
                zeroCount += 1  # 0的个数加1  
        else:  # 如果当前窗口内0的个数超过k  
            if s[l] == '0':  # 如果左边界字符是0  
                zeroCount -= 1  # 0的个数减1  
            l += 1  # 缩小左边界  
    return max_length
```
(3)binary search:最小化max/最大化min，验证函数+二分搜索，将优化问题转化为判断能否达到
**while l\<r; l=mid+1; r=mid(如果mid有可能是答案); 三个变量left,right.ans**
eg.月度开销:分成M个月的最大开销的最小值
```python
def minMaxMonthlyExpense(N, M, expenses):
    def can_split(max_expense):
        #""" 判断是否能合并至多 M 个花费，使最大花费不超过 max_expense """
        months = 1  # 记录当前使用的月份数
        current_sum = 0 # 当前月的开销
        for cost in expenses:
            if current_sum + cost > max_expense:
                months += 1
                if months > M:
                    return False
                current_sum = cost
            else:
                current_sum += cost
        return True
    # 可能的最小开销范围。所以二分是在 [left, right) 区间内进行的
    left, right = max(expenses), sum(expenses) + 1
    ans = -1
    while left < right: # 二分查找最小的 "最大月度开销"
        mid = (left + right) // 2
        if can_split(mid):
            ans = mid   # 记录可行的 `mid`
            right = mid # 继续尝试更小的值
        else:
            left = mid + 1
    return ans
```
(4)区间问题
I.区间合并后的个数：按左端点排序，若与当前无交集则+1，维护右端点
II.选最多的不相交区间/选最少的点使每个区间内都有点：按右端点排序，若与当前无交集则+1，维护右端点
III.选最少区间覆盖已知区间：按左端点排序，选左端点符合要求的区间中右端点最大的一个
```python
# 1. 计算每个位置的覆盖区间
intervals = []
for i, x in enumerate(a, start=1):
	left = max(1, i - x)
	right = min(n, i + x)
	intervals.append((left, right))
# 2. 按左端点排序
intervals.sort()
# 3. 贪⼼选择最少区间覆盖 [1, N]
ans = 0
i = 0
covered = 0
while covered < n:
	best = covered
# 在所有能接上的区间中，选右端点最⼤的
	while i < n and intervals[i][0] <= covered + 1:
		best = max(best, intervals[i][1])
		i += 1
	if best == covered: # ⽆法延伸
		break
	covered = best
ans += 1
```
利用右端最远覆盖数组，O(N)
```python
far = [0] * (n + 2)
for i, x in enumerate(a, start=1):
	L = max(1, i - x)
	R = min(n, i + x)
	far[L] = max(far[L], R)
ans = 0
covered = 0
best = 0
for i in range(1, n + 1):
	best = max(best, far[i])
	if i > covered:
		ans += 1
		covered = best
```
IV.区间分组，最少组使组内区间不相交
```python
# 按左端点从⼩到⼤排序
startEnd.sort(key=lambda x: x[0])
# 创建⼩顶堆
q = []
for i in range(n):
	if not q or q[0] > startEnd[i][0]:
		heapq.heappush(q, startEnd[i][1])
	else:
		heapq.heappop(q)
		heapq.heappush(q, startEnd[i][1])
return len(q)
```
**2.matrix**
(1)保护圈&按一定的方向遍历
eg.螺旋矩阵：撞边拐弯，通过mod4控制方向
```python
s = [[401]*(n+2)]
mx = s + [[401] + [0]*n + [401] for _ in range(n)] + s 
dirL = [[0,1], [1,0], [0,-1], [-1,0]]
row = 1
col = 1
N = 0
drow, dcol = dirL[0]
for j in range(1, n*n+1):
    mx[row][col] = j
    if mx[row+drow][col+dcol]:
        N += 1
        drow, dcol = dirL[N%4]
    row += drow
    col += dcol
```
(2)range中加max，min
eg.垃圾炸弹：存每个点能炸的垃圾数量
```python
d = int(input())  
Moscow = [[0]*1025 for _ in range(1025)]  
n = int(input())  
for k in range(n):  
    x,y,i = map(int,input().split())  
    for p in range(max(0,x-d),min(1025,x+d+1)):  
        for q in range(max(0,y-d),min(1025,y+d+1)):  
            Moscow[p][q] += i  
```
**3.recursion**
(1)backtracking:有很多变量都是可以在递归中传递的，注意base case和invalid case的判断顺序，放入下一个元素可以有一些判断条件（八皇后）
eg.组合总和III
```python
def dfs(l, s, i):#l:当前份数，s:当前和，i:当前最大数
	if l == k and s == n:
		ans.append(sol[:])
		return
	if l > k or s > n or i > 9:
		return
	for j in range(i, 10):
		sol.append(j)
		dfs(l + 1, s + j, j + 1)
		sol.pop()
		dfs(0, 0, 1)
	return ans
```
(2)dfs:通过dfs次数计数，visited防止重复（或者原地修改+回溯）\[马走日&晶矿个数]
eg.晶矿个数
```python
def dfs(grid,i,j,color):  
    if grid[i][j] != f'{color}':  
        return  
    grid[i][j] = '*'  
    d = [[i - 1, j], [i, j - 1], [i, j + 1], [i + 1, j]]  
    for l in range(4):  
        dfs(grid,d[l][0],d[l][1],color)  
  
def ore_count(grid, color):  
    num = 0  
    for p in range(1,n+1):  
        for q in range(1,n+1):  
            if grid[p][q] == f'{color}':  
                dfs(grid,p,q,color)  
                num += 1  
    return num  
  
k = int(input())  
for _ in range(k):  
    n = int(input())  
    m = [['*'] * (n + 2)] + [['*'] + list(input()) + ['*'] for i in range(n)] + [['*'] * (n + 2)]  
    print(ore_count(m, 'r'), ore_count(m,'b'))
```
(3)优化：防止爆栈+记忆化
```python
sys.setrecursionlimit(1<<30)
from functools import lru_cache
@lru_cache(maxsize=None)   #写在函数上面（def上一行）
```
(4)merge sort & quick sort
```python
def MergeSort(arr):#拆成一个一个的，通过merge来sort
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = MergeSort(arr[:mid])
    right = MergeSort(arr[mid:])
    return merge(left, right)
def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result
```
```python
def QuickSort(arr):#取一个作为基准，比ta小放左边，反之放右边
    if len(arr) <= 1:
        return arr
    else:
        pivot = arr[0]  # Choose the first element as the pivot
        left = [x for x in arr[1:] if x < pivot]
        right = [x for x in arr[1:] if x >= pivot]
        return QuickSort(left) + [pivot] + QuickSort(right)
```
(5)disjoint set
I.模板
```python
def find(parent, i):  
    if parent[i] != i:  
        parent[i] = find(parent, parent[i])  
    return parent[i]  
def union(parent, i, j):  
    if i > j:  
        parent[find(parent, i)] = find(parent, j)  
    else:  
        parent[find(parent, j)] = find(parent, i)
```
II.几类题目
i)根节点的个数
```python
len(set(find(representatives,i) for i in range(1,n+1)))
```
ii)按照根节点分组
```python
segment = [[] for _ in range(l)]  
for i in range(l):  
    p = find(rep, i)  
    segment[p].append(s[i])
```
iii)有一些判断，是否矛盾/错的个数：建一个2n+1或者3n+1的rep来表征元素之间的关系（植物观察，食物链）
```python
rep = [_ for _ in range(2*n+1)]  
for k in range(m):  
    p1,p2,con = map(int,input().split())  
    if con == 0:  
        if find(rep,p1) == find(rep,p2+n) or find(rep,p1+n) == find(rep,p2):  
            print("NO")  
            exit()  
        union(rep,p1,p2)  
        union(rep,p1+n,p2+n)  
    else:  
        if find(rep,p1) == find(rep,p2) or find(rep,p1+n) == find(rep,p2+n):  
            print("NO")  
            exit()  
        union(rep,p1,p2+n)  
        union(rep,p1+n,p2)
```
**4.dynamic programming**
(1)状态转移方程
i)定义状态时，当前状态只能涉及之前计算过的值，不知道之前具体的状态（所以最长上升子列要二重循环）。解决这一问题：**（如果状态/操作比较少）多个内存空间**/状态压缩（多条路径）
ii)如何写：定义状态，对当前的状态分类讨论（整数划分：有/无1；卡特兰数：1出栈的位置），用之前算过的状态表示
iii)注意最后要输出的是dp\[n]还是max(dp)
(2)滚动数组，Kadane算法
```python
def kadane(arr):
	max_current = max_global = arr[0]
	for i in range(1, len(arr)):
		max_current = max(arr[i], max_current + arr[i])
		max_global = max(max_global, max_current)
	return max_global
```
eg.不降数组：dp+prefix sum+滚动数组+bisect
```python
from bisect import bisect_right  
def solve():  #注意要有return，不要在函数里print就完了，可能会RE
    n, m = map(int, input().split())  
    matrix = [list(map(int, input().split())) for _ in range(n)]  
    dp = [[0] * m for _ in range(2)]  
    for i in range(m):  
        dp[0][i] = 1  
    for i in range(1,n):  
        prev_row = (i-1) % 2  
        curr_row = i % 2  
        prefix_sum =[0] * (m+1)  
        for j in range(m):  
            prefix_sum[j+1] = prefix_sum[j] + dp[prev_row][j]  
        for k in range(m):  
            dp[curr_row][k] = 0  
        for k in range(m):  
            r = bisect_right(matrix[i-1], matrix[i][k])  
            if r > 0:  
                dp[curr_row][k] = prefix_sum[r]  
    last_row = (n-1) % 2  
    ans = sum(dp[last_row])  
    return ans  
print(solve())
```
(3)背包dp
I.0-1背包：每个物品选/不选，只有一个。eg.小偷背包dp\[i]\[j]:前i个物品，背包容量j的最大价值，状态转移：不能取/能取第i个物品；取/不取第i个物品
一维：`dp[j]` 实际上表示 `dp[i][j]`，j要倒着遍历
```python
def zero_one_pack(V, v, w):
    """
    0-1背包问题
    V: 背包容量
    v: 物品体积列表
    w: 物品价值列表
    """
    n = len(v)
    dp = [0] * (V + 1)
    
    for i in range(n):  # 遍历物品
        # 关键：逆序遍历容量
        for j in range(V, v[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - v[i]] + w[i])
    
    return dp[V]
```

II.完全背包：每个物品可以取0-无限个，dp\[i]=max(dp\[i-w\[i]]+w\[i]/1,dp\[i])，状态转移方程和一维的0-1背包完全一样，但是是正着遍历
III.多重背包：将多个相同的物品看成不同的，从而化为0-1背包
IV.恰好背包：0设为0，其余设为-1/-inf，初值不为-1/-inf才更新
eg.健身房
```python
t,n=map(int,input().split())  
dp=[0]+[-1]*(t+1)  
for i in range(n):  
    k,w=map(int,input().split())  
    for j in range(t,k-1,-1):  
        if dp[j-k]!=-1:  
            dp[j]=max(dp[j-k]+w,dp[j])  
print(dp[t])
```
(4)其他
I.上三角dp(True/False)：最长回文子串，dp\[i]\[j]：s\[i:j+1]是否回文
II.二维dp：最长公共子列(连续)/最长公共子串
III.技巧：多个dp数组；**左右各扫一遍**（登山，摆动序列）
**5.searching**
(1)bfs
I.模板：以迷宫最短路径为例
```python
def bfs():  
    inq = set()  #visited
    inq.add((1,1))  
    q = deque()  
    q.append((0,1,1))  #初始化，最好就分4步写
    while q:  
        step,x,y=q.popleft()  
        if x==n and y==m: #终止条件
            return q  
        for dx,dy in [(0,1),(0,-1),(1,0),(-1,0)]:  
            nx,ny=x+dx,y+dy #下一个入队的节点
            if 0<=nx<n+2 and 0<=ny<m+2 and maze[nx][ny]==0: #is_valid 
                if (nx,ny) not in inq:  
                    inq.add((nx,ny))  
                    path[nx][ny]=(x,y) #为了后续通过前驱点找路
                    q.append((step+1,nx,ny))  
    return None
```
II.可以传递的变量很多：比如当前的状态（蛇入迷宫，小游戏：蛇横/竖），或者到达的路径（跳房子），也可以通过前驱点还原出路径（如下）
```python
def print_path(path,pos):  
    res = []  
    while pos!=(-1,-1):  
        res.append(pos)  
        pos=path[pos[0]][pos[1]]  
    res.reverse()
```
(2)dijkstra:用heapq, heappop, heappush代替deque, popleft, append，保证最先弹出的是最小的，注意要先把图构建出来（subway）
eg.走山路
```python
def dijkstra(start_x, start_y, end_x, end_y):  
    pos = []  
    dist = [[float('inf')] * n  for i in range(m)]  
    if terrain[start_x][start_y] == "#":  
        return 'NO'  
    dist[start_x][start_y] = 0  
    heapq.heappush(pos, (0, start_x, start_y))  
    while pos:  
        d, x, y = heapq.heappop(pos)  
        if d > dist[x][y]:  
            continue  
        if x == end_x and y == end_y:  
            return d  
        h = int(terrain[x][y])  
        for dx, dy in directions:  
            new_x, new_y = x + dx, y + dy  
            if 0 <= new_x < m and 0 <= new_y < n and terrain[new_x][new_y] != "#":  
                if d + abs(int(terrain[new_x][new_y]) - h) < dist[new_x][new_y]:  
                    dist[new_x][new_y] = d + abs(int(terrain[new_x][new_y]) - h)  
                    heapq.heappush(pos, (dist[new_x][new_y], new_x, new_y))  
    return 'NO'
```
**6.数据结构**
(1)heapq:快速找最大/最小值，会比较占内存，虽然每次加新元素都要heappush但尽量只在必要的地方heappush，像辅助栈一样也有双堆
(2)queue:除了在bfs用就是约瑟夫问题了
(3)stack
I.一般的stack：括号匹配eg.咒语序列（往stack里存idx，enumerate函数）
```python
s = input()  
stack = [-1]  
max_len = 0  
for i, c in enumerate(s):  
    if c == '(':  
        stack.append(i)  
    else:  
        stack.pop()  
        if not stack:  
            stack.append(i)  
        else:  
            current_len = i - stack[-1]  
            if current_len > max_len:  
                max_len = current_len  
print(max_len)
```
判断括号是否有效("".join(str))
```python
for x in ans:
	stack = []
	sieve = True
	for y in x:
		if y == "(":
			stack.append(y)
		else:
			if not stack:
				sieve = False
				break
			stack.pop()
	if not stack and sieve:
		res.append("".join(x))
```
II.monotonic stack
i)基础：eg.寻找下一个更大元素
```python
def monotonic_stack_template(nums):
    n = len(nums)
    stack = []  # 通常存储索引而不是值
    result = [默认值] * n  # 存储结果
    
    for i in range(n):
        # 当栈不为空且当前元素满足某种条件时，弹出栈顶元素
        while stack and nums[i] > nums[stack[-1]]:  # eg.单增单调栈
            idx = stack.pop()
            result[idx] = nums[i]  # 或 i - idx 等，根据具体问题
        
        # 将当前索引入栈
        stack.append(i)
    
    # 处理栈中剩余元素（如果有需要）
    while stack:
        idx = stack.pop()
        result[idx] = 默认值  # 如 -1 表示没有找到
    
    return result
```
ii)矩形：eg.护林员盖房子又来了：柱状图最大矩形+二维变一维（最大子矩阵也有同样的思想）
```python
def max_rectangle_area(heights):  
    stack = []  
    heights = [0] + heights + [0]  
    ans = 0  
    for i in range(m+2):  
        while stack and heights[stack[-1]] > heights[i]:  
            l = heights[stack.pop()]  
            w = i - stack[-1] - 1  
            ans = max(ans, l * w)  
        stack.append(i)  
    return ans  
def max_matrix():  
    h = [0] * (m+1)  
    ans = 0  
    for row in forest:  
        for j, c in enumerate(row):  
            if c == 1:  
                h[j] = 0  
            else:  
                h[j] += 1  
        ans = max(ans, max_rectangle_area(h))  
    return ans
```
iii)贡献法:eg.子数组最小乘积（以每个数为中心找以ta为最小值的最大子数组+前缀和\[下标结合例子来写]）
```python
def maxSumMinProduct(self, nums) -> int:  
    nums = [0] + nums + [0]  
    pre = [0]  
    for n in nums:  
        pre.append(pre[-1]+n)  
    right = [None] * len(nums)  
    stack = []  
    for i in range(len(nums)):  
        while stack and nums[i] < nums[stack[-1]]:  
            right[stack.pop()] = i  
        stack.append(i)  
    left = [None] * len(nums)  
    stack = []  
    for i in range(len(nums)-1,-1,-1):  
        while stack and nums[i] < nums[stack[-1]]:  
            left[stack.pop()] = i  
        stack.append(i)  
    ans = 0  
    for i in range(1,len(nums)-1):  
        l = left[i]  
        r = right[i]  
        ans = max(ans, nums[i]*(pre[r]-pre[l+1]))  
    return ans
```
III.auxiliary stack：用另一个栈同步存需要的东西
eg1.快速堆猪：另一个栈存最小猪，用最小猪确保个数相同
eg2.今日化学论文：通过两个栈把倒序弹出的字符顺成正的(\*,sep="")
```python
s = input()  
stack = []  
for i in range(len(s)):  
    stack.append(s[i])  
    if stack[-1] == "]":  
        stack.pop()  
        helpstack = []  
        while stack[-1] != "[":  
            helpstack.append(stack.pop())  
        stack.pop()  
        numstr = ""  
        while helpstack[-1] in "0123456789":  
            numstr += str(helpstack.pop())  
        helpstack = helpstack*int(numstr)  
        while helpstack != []:  
            stack.append(helpstack.pop())  
print(*stack, sep="")
```
**7.一些技巧**
(1)prefix sum(二维)
(0,0)到(i-1,j-1):prefix\[i]\[j]=matrix\[i-1]\[j-1]+prefix\[i-1]\[j]+prefix\[i]\[j-1]-prefix\[i-1]\[j-1]
(x1,y1)到(x2,y2):s=prefix\[x2+1]\[y2+1]-prefix\[x1]\[y2+1]-prefix\[x2+1]\[y1]+prefix\[x1]\[y1]
(2)用字典统计个数的时候要把题目的意思转化成“键的性质”，不要跟太多数相关
eg.4数和为0的元组总数（还有same difference)
```python
dict1 = {}
for i in range(n):
	for j in range(n):
		if not a[i]+b[j] in dict1:
			dict1[a[i] + b[j]] = 0
			dict1[a[i] + b[j]] += 1
ans = 0
for i in range(n):
	for j in range(n):
		if -(c[i]+d[j]) in dict1:
			ans += dict1[-(c[i]+d[j])]
```
**8.语法**
(1)**enumerate(索引+元素)和zip(并行迭代多个)**    eg.子数组最小数之和（单调栈贡献法）
```python
class Solution:  
    def sumSubarrayMins(self, arr) -> int:  
        n = len(arr)  
        st_l = []  
        left = [-1]*n   #数组之外的位置  
        for i,x in enumerate(arr):  
            while st_l and arr[st_l[-1]] >= x:  #两边有一边取等一边不取，确保每个元素都取到  
                st_l.pop()  
            if st_l:  
                left[i] = st_l[-1]  
            st_l.append(i)  
        st_r = []  
        right = [n]*n  
        for i in range(n-1,-1,-1):  
            while st_r and arr[st_r[-1]] > arr[i]:  
                st_r.pop()  
            if st_r:  
                right[i] = st_r[-1]  
            st_r.append(i)  
        ans = 0  
        for i,(x,l,r) in enumerate(zip(arr,left,right)):  
            ans += x*(i-l)*(r-i)  
            ans = ans % (10**9+7)  
        return ans% (10**9+7)
```
(2)any()和all()  
eg.```all(x>10 for x in a)-->True/False```
(3)map()和filter()+lambda x
eg.```squared_numbers_iterator = map(lambda x: x**2, numbers)
even_numbers_iterator = filter(lambda x: x % 2 == 0, numbers)
nums.sort(key=lambda x: (x[0],-x[1]), reverse=True)```
sort（原地修改）和sorted（返回新的列表）
(4)对列表：reduce()和reverse()
eg.```sum_result = reduce(lambda x, y: x + y, numbers)
"".join(reversed(str/list))```
(5)str.translate(mapping)和str.maketrans('被替换的‘，’新的‘)
eg.```trans_table = str.maketrans('aeiou', '*****')```
(6)join()和字典
```''.join(map(str,s))``` 将列表s中的元素⽆间隔连接
```keys=[key for key,value in d.items() if value==target] ```根据值找到所有的键
```key=next((key for key,value in d.items() if value==target),None)``` 根据值找到第⼀个对应的键
pow(base,exp,mod)
(7)埃氏筛及筛法
```python
    sieve = [True]*2001  
    sieve[0] = False  
    sieve[1] = False  
    for i in range(2,int(2000**0.5)+1):  
        if sieve[i]:  
            for j in range(i*i,2001,i):  
                sieve[j] = False  
    primes = [i for i in range(2,2000) if sieve[i]]  
```
筛法eg.求亲和数：遍历因子得到各个数的因子之和
```python
n = int(input())  
s = [1]*n  
ans = []  
for i in range(2,n//2+1):  
    k = 2  
    while k * i <= n:  
        s[k*i-1] += i  
        k += 1  
for i in range(n):  
    if i+1<s[i]<n:  
        if s[s[i]-1]==i+1:  
            print(i+1,s[i])
```
