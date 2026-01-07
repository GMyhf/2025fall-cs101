---
tags:
  - note
---
# 一些小点
## 1.有关删除：
除非是pop等能明确查找的操作，否则不要轻易使用删除，时间复杂度为O(n),还会带来内存方面的压力，一般采用标记或懒删除等方法
## 2.字典相关：
- dict.get(key， 默认值):不存在不会报错，而是取默认值，没有默认值就取None
- dict\[key]:不存在会报错
- dict.setdefault(key, value):key不存在则创建，存在则更新
- dict.update(dictt):合并两字典
- 善用defaultdict
- 迭代时不要直接修改字典（一般也不会修改），如果执意要修改，先复制
- 更pythonic的写法：for k, v in dict.items():
## 3.字符串相关：
- 字符串可直接比较大小（字典序），注意大小写的问题
- isspace(): 只包含空格返回True
- isalnum()：至少有一个字符且全为字母或数字则返回True
- isalpha()：至少有一个字符且全为字母返回True
- isdecimal()：只包含数字返回True（全角数字）
- istitle()：每个单词首字母大写返回True
- islower()：至少包含一个区分大小写的字符且全为小写返回True（isupper类似）
- title()：每个单词首字母大写
- lower()：大写变小写（upper类似)
- swapcase()：翻转大小写
- startswith(str)：以str开头返回True（endswith类似)
- find(str, start=0, end=len(string))：检查str是否在指定范围中，如果是则返回开始的索引值，否则返回-1（大多数情况下快于KMP）（rfind函数从右开始查找）（index类似，只是不存在时会报错）（另有rindex)
- replace(old_str, new_str, num=string,count(old))：替换字符串，可指定次数，不指定则全部替换，不改变原字符串
- strip()：截掉左右的空白字符（类似又lstrip、rstrip）
- ljust(width)：返回一个原字符串左对齐，并使用空格填充至width的新字符串（类似有rjust、center）
- partition(str)：把字符串分成一个三元组（str之前、str、str之后）（rpartition从右查找）
- split(str='', num)：以str为分割符拆分字符串，若num指定，仅仅分割出`num + 1`个字符串，str默认包含'\r', '\n', '\r\n'和空格，返回列表
- splitlines()：按行分割，返回列表
- join(seq)：以字符串作为分隔符，将seq中所有元素（的字符串表示）合并为一个新字符串
- ord返回ASCII码，chr还原
## 4.math模块：
- hypot()：取模、ceil()：上取整、factonal()：阶乘、gcd()：最大公因数、trunc()：去掉小数
# 数据结构
## 1.并查集（Disjoint-Set)
### 1.基本情况：
- 一些互不相交的集合，每个集合可以用一个树来表示，其中树的根节点是集合的代表元素
### 2.操作：
- 添加新元素
- union：合并集合
- find：找到根节点（代表元素）
- 检查是否合法（两集合是否不相交）
- 初始化：一般每个元素最初属于不同集合
### 3.树的一些细节：
- 用一个parent数组来实现，其第i项是第i个元素的父节点
- 若parent\[i] = i，则i即为根节点，查找停止
### 4.本质：动态维护等价关系的工具
### 5.代码实现：
#### 1.初始化：
```python
parent = list(range(n))
```
#### 2. find：
```python
def find(i):
	if parent[i] == i:
		return i
	"""为根节点，返回"""
	return find(parent[i])
	"""不为根节点，递归查找"""
```
#### 3.union：
```python
def union(x, y):
	px, py = find(x), find(y)
	parent[px] = py
	"""将x集合的代表元素变为y，即合并"""
```
### 6.优化：
#### 1.union by rank:
- 准备一个新数组rank,用来记录每棵树的高度（秩）
- 由于union操作时的对称性，将哪一棵树移动并不重要，现在我们希望最小化树的高度（来减少find操作的时间复杂度和栈的深度），将秩更小的树移到另一个下方
- 记得维护rank列表中的值
#### 2.path compression
- 利用find方法，每次查找都将子节点往根节点连接，最大程度地减少树的高度
#### 3.union by size:
- 与union by rank类似，只不过比较方法变成了广度优先，不再赘述
#### 4.代码实现：
```python
class dis_set:
	def __init__(self, n):
		self.rank = [1] * n
		self.parent = list(range(n))
	
	def find(self, x):
		if self.parent[x] != x:
			self.parent[x] = self.find(self.parent[x])
			"""path compression"""
		return self.parent[x]
	
	def union(self, x, y):
		px, py = self.find(x), self.find(y)
		if px == py:
			return
		"""union by rank"""
		if self.rank[px] < self.rank[py]:
			self.parent[px] = py
		elif self.rank[px] > self.rank[py]:
			self.parent[py] = px
		else:
			self.parent[py] = px
			self.rank[px] += 1
```
### 7.典例：
#### 1.如果树有颜色：
- 不同集合间的等价关系削弱时可引入新数组来描述每个集合的颜色

M27306 植物观察

周末，小明和他的朋友们去野外观察植物，并拍了 n 张植物的照片回来。社团的前辈告诉他们，照片上所有这些植物都属于两个不同种类 A、B 中的某一种。由于是新手，小明很难直接判断每张照片上的植物属于哪个种类。但是，如果把植物的照片两两组合对比来看，小明能够判断其中一些组合的两个植物是属于相同还是不同的种类。经过一番研究，小明最终得出了 m 组这样的结论。他希望知道，这 m 组结论是否可能同时成立。具体来说，是否存在一种给每个植物分类为 A 或 B 的方法，使得小明的每组结论对于两个植物种类相同或不同的判断均正确。

输入：第一行为两个整数，分别是植物的数量 n 和结论的个数 m。n <= 100, m <= 10000，接下来 m 行，每行为一个结论，包括三个整数，前两个整数表示植物的编号（从 0 到 n-1），第三个整数表示小明的判断，0 代表相同，1 代表不同。

输出：如果所有结论可能同时成立，输出 “YES”，否则输出 “NO”。

```python
def find(x):  
    if parent[x] != x:  
        op = parent[x]
        """缓存父节点"""
        root = find(op)  
        parity[x] ^= parity[op] 
        """parity数组也隐含地递归维护，每一个节点与它的父节点异或运算（相同为0不同为1），最终         达到描述任意节点与根节点的组关系的效果"""
        parent[x] = root  
        """压缩路径"""
    return parent[x]  
  
  
def union(x, y, w):  
	"""这里union还兼顾了判断是否合法的任务"""
    rx, ry = find(x), find(y)  
    px, py = parity[x], parity[y]  
    if rx == ry:  
        return (px ^ py) == w  
        """处于同一组，px与py有相同的参考依据（同一个根节点），（px ^ py）即可反映x与y的类别         关系，再^上w即可判断是否合法"""
    if rank[rx] < rank[y]:  
        parent[rx] = ry  
        """按秩合并"""
        parity[rx] = px ^ py ^ w  
        """在本例中两者只要有关系（不管是已知同组还是异组）都会被划分到同一集合，因此需要合并集
        合时的组类情报（w）都是暂时为真的，w反映了xy的组类关系，px反映x与父节点，py类似，合一         起就反映了两根节点的组类关系，对被合并的根节点更新即可（后续将在find中更新各子节点，递         归地相邻地更新来确保不会有误"""
    else:  
        parent[ry] = rx  
        parity[ry] = px ^ py ^ w  
        if rank[rx] == rank[ry]:  
            rank[rx] += 1  
    return True  
  
  
n, m = map(int, input().split())  
parent, parity, rank = list(range(n)), [0] * n, [0] * n  
for _ in range(m):  
    a, b, c = map(int, input().split())  
    if not union(a, b, c):  
        print('NO')  
        break  
else:  
    print('YES')
```

注释：我们暂且忽视了植物不同类的特性使得集合能够对称地分割，再用parity数组来描述子节点与父节点（维护之后为根节点）的分组关系，巧妙地运用了异或运算
## 2.单调栈（单调队列）：
### 1.定义：
单调栈是一种特殊的栈结构，其中的元素按照某种特定的顺序（如递增或递减）排列。
### 2.应用场景：
- 1. **寻找下一个更大（小）的元素**：给定一个数组，对于每个元素，找到它右边第一个比它大的元素的位置。这类问题可以使用单调递减栈来高效解决。
- 2. **直方图中的最大矩形**：这是一个经典的问题，涉及到计算直方图中最大的矩形面积，可以使用单调栈来有效求解。
### 3.原理：
- **入栈操作**：当一个新的元素需要加入到栈中时，根据栈的性质（递增或递减），将所有不符合条件的栈顶元素弹出，然后再将新元素压入栈中。
- **出栈操作**：通常情况下，出栈操作是自动发生的，即在执行入栈操作时，为了保持栈的单调性，会自动移除不满足条件的栈顶元素。
### 4.典例：
#### 1.找到数组中每个元素右边第一个更大的数
在第一次遇到比它大的数时，该元素出栈，后续不被操作，最终得到一个递减的单调栈
```python
def next_greater_element(nums):  
    stack = []  
    result = [0] * len(nums)  
    for i in range(len(nums)):  
        """当栈不为空且当前考察的元素大于栈顶元素时"""
        while stack and nums[i] > nums[stack[-1]]:  
            index = stack.pop()  
            result[index] = nums[i]  
        """将当前元素的索引压入栈中""" 
        stack.append(i)  
    """对于栈中剩余的元素，它们没有更大的元素"""  
    while stack:  
        index = stack.pop()  
        result[index] = -1  
    return result  
  
num = [4, 5, 2, 25]  
print(next_greater_element(num))
```
#### 2.直方图中最大矩形：
关键：找到每个柱子左右两边最近的高度比自己小的柱子即可，就转化为了上一题，维护两个单调栈即可
```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        n = len(heights)
        left, right = [0] * n, [0] * n
        
        stack = []
        for i in range(n):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            left[i] = stack[-1] if stack else -1
            stack.append(i)

        stack = []
        for i in range(n - 1, -1, -1):
            while stack and heights[stack[-1]] >= heights[i]:
                stack.pop()
            right[i] = stack[-1] if stack else n
            stack.append(i)

        res = max((right[i] - left[i] - 1) * heights[i] for i in range(n))
        return res
```
## 3.前缀树：
### 1.基本信息：
字典树是一种树形数据结构，用于高效地存储和检索字符串数据集中的键。如果你使用嵌套的字典来表示字典树，其中每个字典代表一个节点，键表示路径上的字符，而值表示子节点，那么就构成了字典树。一般用字典实现，所以也叫字典树
### 2.用途：
用于高效地存储和检索字符串数据集中的键。这一数据结构有相当多的应用情景，例如自动补全和拼写检查。
### 3.代码实现：
```python
class Trie:
    def __init__(self):
        """
        初始化前缀树（Trie）
        使用嵌套字典结构存储节点
        每个节点是一个 dict，键为字符
        特殊键 "#" 表示一个单词的结束位置
        """
        self.root = {}             """Trie 的根节点，用字典表示"""
        self.end_of_word = "#"     """特殊标记：表示该路径为一个单词的结束"""

    def insert(self, word: str) -> None:
        """将一个单词插入 Trie"""
        node = self.root
        for char in word:
            """setdefault：若 char 存在返回已有节点；
            若不存在则创建一个空 dict 作为子节点并返回它"""
            node = node.setdefault(char, {})
        """在单词末尾添加结束标记"""
        node[self.end_of_word] = self.end_of_word

    def search(self, word: str) -> bool:
        """判断一个完整单词是否在 Trie 中"""
        node = self.root
        for char in word:
            """如果任意字符不存在，说明该单词不存在"""
            if char not in node:
                return False
            node = node[char]
        """需要检查结束标记，确保是完整单词而不是前缀"""
        return self.end_of_word in node

    def startsWith(self, prefix: str) -> bool:
        """判断是否存在以 prefix 为前缀的单词"""
        node = self.root
        for char in prefix:
            """若路径中缺字符，则没有这个前缀"""
            if char not in node:
                return False
            node = node[char]
        """只需前缀完整，无需检查结束标记"""
        return True
```
## 4.矩阵：
由于用的比较多，这里仅列出相关数学性质的实现,注意看时机添加保护圈即可：
```python
class Matrix:
	def __init__(self, data):
		self.data = data
		self.rows = len(data)
		self.cols = len(data[0])
		
	def __matmul__(self, other): """矩阵乘法"""
		if self.cols != other.rows:
			raise ValueError('Matrix dimensions do not match for multiplication')
		result = [[0] * other.cols for _ in range(self.rows)]
		for i in range(self.rows):
			for j in range(other.cols):
				for k in range(self.cols):
					result[i][j] += self.data[i][k] * other.data[k][j]
		return Matrix(result)
		
	def __add__(self, other): """矩阵加法"""
		if self.rows != other.rows or self.cols != other.cols:
			raise ValueError('Matrix dimensions do not match for addition')
		result = [
			[self.data[i][j] + other.data[i][j] for j in range(self.cols)]
			for i in range(self.rows)
		]
		return Matrix(result)
		
	def __str__(self): """打印友好"""
		return '\n'.join(' ' .join(map(str, row)) for row in self.data)
```
# 滑动窗口、双指针：
## 1.思想：
- 若（l, r）能满足条件 => （l, r + 1）~（l, end）均满足条件，则以l为起点的不用再次遍历，从（l + 1）开始即可
- 若（l, r）不满足条件 => （l + 1, r）~（r - 1, r）不满足，则r为终点不用再遍历，从（r + 1）开始即可
## 2.典例：
给你一个数组 `nums` 和一个整数 `k`，请输出每个长度为 `k` 的子数组的最大值（子数组必须相连）
思想：
- 双指针维护窗口，deque优化维护
- 一个递减的单调栈维护最值，当右端加入新元素时，把所有比它小的元素弹出（因为它们之后不会成为最大值）
- 当左端元素滑出窗口时，如果正好是队首，就弹出
```python
from collections import deque
nums = [1,3,-1,-3,5,3,6,7]
k = 3

def maxSlidingWindow(nums, k):
    dq = deque([])  """存下标，保证对应值递减"""
    res = []
    for right, x in enumerate(nums):
        """step 1: 窗口右扩，保持单调递减"""
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(right)

        """step 2: 移除滑出窗口的左端元素"""
        if dq[0] <= right - k:
            dq.popleft()

        """step 3: 当窗口形成（长度 >= k）时，记录最大值"""
        if right >= k - 1:
            res.append(nums[dq[0]])
    return res

print(maxSlidingWindow([1,3,-1,-3,5,3,6,7], 3))
```
## 3.KMP:
### 1.用途：
字符串匹配
### 2.思想：
动态规划，双指针，底层逻辑比较复杂，会用即可
### 3.代码：
```python
def build_next(patt):
	next = [0]
	prefix_len = 0 """当前共同前后缀的长度"""
	i = 1
	while i < len(patt):
		if patt[prefix_len] ==  patt[i]:
			prefix_len += 1
			next.append(prefix_len)
			i += 1
		else:
			if prefix_len == 0:
				next.append(0)
				i += 1
			else:
				prefix_len = next[prefix_len - 1]
		return next
		
def kmp(string, patt):
	next = build_next(patt)
	i, j = 0, 0
	while i < len(string):
		if string[i] == string[j]:
			i += 1
			j += 1
		elif j > 0:
			j = next[j - 1]
		else:
			i += 1
		if j == len(patt):
			return i - j
```
## 4.马拉车：
### 1.用途：
判断最长回文子串
### 2.思想：
动态规划+双指针，原理就是利用对称，底层逻辑较复杂，会用即可
```python
def expend(s, left, right):
	while left >= 0 and right < len(s) and s[left] == s[right]:
		left -= 1
		right += 1
	return (right - left - 2) // 2
	
def manacher(s):
	end, start = -1, 0
	s = '#' + '#'.join(list(s)) + '#'
	arm_len = []
	right, j = -1, -1
	for i in range(len(s)):
		if right >= i:
			i_sym = 2 * j - 1
			min_arm_len = min(arm_len[i_sym], right - i)
			cur_arm_len = expend(s, i - min_arm_len, i + min_arm_len)
		else:
			cur_arm_len = expend(s, i, i)
		arm_len.append(cur_len_arm)
		if i + cur_arm_len > right:
			j = i
			right = i + cur_arm_len
		if 2 * cur_arm_len + 1 > end - start:
			start = i - cur_arm_len
			end = i + cur_arm_len
	return s[start + 1:end + 1:2]
```
# 二维前缀和
## 1.定义：
prefix\[i]\[j]表示从(0, 0)到(i - 1, j - 1)求和
## 2. 计算公式：
prefix\[i]\[j] = matrix\[i - 1]\[j - 1] + prefix\[i - 1]\[j] + prefix\[i]\[j - 1] - prefix\[i - 1]\[j - 1]
## 3.查询(x1, y1) ~ (x2, y2) 的面积：
sum = prefix\[x2 + 1]\[y2 + 1] - prefix\[x1]\[y2 + 1] - prefix\[x2 + 1]\[y1] + prefix\[x1]\[y1]
# 二分查找
## 1.基本信息：
适用于较大有序数据集,原理就是二分法
## 2.基本代码：
```python
def binary_search(arr, target):
	left, right = 0, len(arr) - 1
	while left <= right:
		mid = (left + right) // 2
		if arr[mid] == target:
			return mid
		elif arr[mid] < target:
			left = mid + 1
		else:
			right = mid - 1
	return -1
```
## 3.标准库：
当然，这么常使用的算法肯定是被python标准库收集了，只需导入bisect库的bisect_left或bisect_right
不过源码值得我们鉴赏
```python
def bisect_left(a, x, lo=0, hi=None, *, key=None):
 """Return the index where to insert item x in list a, assuming a is sorted.

 The return value i is such that all e in a[:i] have e < x, and all e in
 a[i:] have e >= x.  So if x already appears in the list, a.insert(i, x) will
 insert just before the leftmost x already there.

 Optional args lo (default 0) and hi (default len(a)) bound the
 slice of a to be searched.

 A custom key function can be supplied to customize the sort order.
 """

 if lo < 0:
     raise ValueError('lo must be non-negative')
 if hi is None:
     hi = len(a)
 """Note, the comparison uses "<" to match the
 __lt__() logic in list.sort() and in heapq."""
 if key is None:
     while lo < hi:
         mid = (lo + hi) // 2
         if a[mid] < x:
             lo = mid + 1
         else:
             hi = mid
 else:
     while lo < hi:
         mid = (lo + hi) // 2
         if key(a[mid]) < x:
             lo = mid + 1
         else:
             hi = mid
 return lo
 ```
- bisect_left(a, x, lo=0, hi=len(a), \*, key=None) 返回将 x 插入到列表 a 的**最左位置**（如果 x 已存在，返回其左侧）。适合查找“小于 x 的右侧边界”。
- bisect_right(a, x, lo=0, hi=len(a), \*, key=None)（或 bisect(a, x, ...)） 返回将 x 插入到列表 a 的**最右位置**（如果 x 已存在，返回其右侧）。
## 4.应用：
### 1.最大上升子序列（不要求连续）：
#### 1.原理：
维护一个dp数组，用来记录长度为 i+1 的递增子序列的最小末尾元素（经典 LIS 优化算法中的辅助数组）。最后找到已经维护过的最长数组即可
#### 2.代码：
```python
from bisect import bisect_left  
n = int(input())  
nums = list(map(int, input().split()))  
dp = [1e9] * n  
for i in nums:  
"""dp每一步都是通过bisect构造的，因此天然满足单调性，不用再排序"""
    dp[bisect_left(dp, i)] = i
    """若当前元素大于dp中某一个元素，可以更新，更新前面的其实对之后的答案没有影响"""
print(bisect_left(dp, 1e9))
```
#### 3.注意：
上面的代码是对于严格递增的子列来说的，若是不严格递增，用bisect_right即可
### 2.缩小解的范围：
对于数据连续分布且范围已知（或大概已知）的题目，用二分查找可以快速确定解的范围，重复下去就能得到最终答案
一个典例：nums表示第i个袋子里球的数目，m表示操作最大执行数，操作为选一个袋子将球分到2个新袋子中（不能为空），现在要求出球数最大值的最小值
```python
left, right, res = 1, max(nums), 0
"""左右指针框定二分查找范围"""
while left <= right:
	y = (left + right) // 2
	ops = sum((x - 1) // y for x in nums) """每个袋子的最少操作数"""
	if ops <= m:
		res = y
		right = y - 1
	else:
		left = y + 1
```
# 动态规划
## 1.状态压缩：
### 1.意义：
众所周知dp中最核心的就是状态，有时候状态不能简单的用二维数组来表达，为了避免高维数组的使用，我们使用状态压缩，用一个数来表示状态，从而能使二维数组可以使用，而这种状态的表示往往采用二进制及位运算来表示
### 2.位运算：
- <<：左移，1 << v 表示把1向左移动v位
- &：按位与，mask1 & mask2
- |：按位或，mask1 | mask2
- ^:按位异或，mask1 ^ mask2
- ~：按位取反，~mask
### 3.常见操作：
- 判断第i位是不是1:
```python
(state >> i) & 1
```
- 把第i位变成1:
```python
state |= (1 << i)
```
- 把第i位变成0：
```python
state &= ~(1 << i)
```
- 反转第i位（选或不选切换状态）：
```python
state ^= (1 << i)
```
- 枚举所有子集
```python
def subsets(state)
	sub = state
	while True:
		yield sub """把函数变成一个生成器"""
		if sub == 0:
			break
		sub = (sub - 1) & state
```
### 4.一个例子——TSP：
```python
n = int(input())  
cost = [list(map(int, input().split())) for _ in range(n)]  
INF = float('inf')  
dp = [[INF] * n for _ in range(1 << n)]  
dp[1][0] = 0 for city in range(1 << n):  
    for u in range(n):  
        if dp[city][u] >= INF:  
            continue  
        for v in range(n):  
            if city & (1 << v):    
                continue  
            new_mask = city | (1 << v)  
            dp[new_mask][v] = min(dp[new_mask][v], dp[city][u] + cost[u][v])  
ans = INF  
full_mask = (1 << n) - 1  
for i in range(n):  
    if dp[full_mask][i] < INF:  
        ans = min(ans, dp[full_mask][i] + cost[i][0])    
print(int(ans) if ans < INF else -1)
```
## 2.Kadane算法：
### 1.应用场景：
最大子数组和问题，即在一个整数数组中找到连续子数组的最大和
### 2.代码实现：
```python
def kadane(arr):
    curr_max = total_max = arr[0]
    for x in arr[1:]:
        curr_max = max(x, curr_max + x)  """要么重新开始，要么接上前面"""
        total_max = max(total_max, curr_max)
    return total_max
```
### 3.扩展到二维——最大子矩阵：
#### 1.总体策略：
1. 枚举所有可能的**上边界 `top`**
2. 对每个 `top`，枚举所有 `bottom >= top`
3. 对每一对 `(top, bottom)`，计算从第 `top` 行到第 `bottom` 行的**每列的累加和**，形成一个一维数组 `col_sum`
4. 在 `col_sum` 上运行 Kadane 算法，得到当前上下边界下的最大子矩阵和
5. 更新全局最大值
#### 2.代码实现：
```python
from sys import stdin  
N = int(input())  
data = iter(stdin.read().strip().split())  
ma = [[int(next(data)) for _ in range(N)] for _ in range(N)]  
res = te = 0  
for i in range(N):  
    for j in range(i, N):  
        te = 0  
        for t in range(i, j + 1):  
            te += ma[t][0]  
        for k in range(1, N):  
            temp = 0  
            for t in range(i, j + 1):  
                temp += ma[t][k]  
            te = max(temp, temp + te)  
            res = max(res, te)  
print(res)
```
## 3.背包问题：
### 1.0-1背包（每个物品选或不选）：
#### 1.状态转移方程：
 CELL\[i]\[j] 表示前 i 件物品恰放入一个容量为 j 的背包可以获得的最大价值。则其状态转移方程便是： CELL\[i]\[j]=max(CELL\[i−1]\[j];CELL\[i−1]\[j−Wi]+Vi)（绝大多数背包问题都是以这个状态转移方程为基础）
#### 2.基本代码：
```python
n, b = map(int, input().split())  
price = list(map(int, input().split()))  
weight = list(map(int, input().split()))  
dp = [[0] * n for _ in range(b + 1)]  
for i in range(b + 1):  
    for j in range(n):  
        if i - weight[j] >= 0:  
            dp[i][j] = max(dp[i][j - 1], dp[i - weight[j]][j - 1] + price[j])  
print(dp[-1][-1])
```
#### 3.优化——滚动数组：
由于上面的状态转移方程表示前件物品的递推只涉及相邻两项，可以用滚动数组减去一个维度，节省空间，即：
```python
for i in range(n):
    for l in range(W, w[i] - 1, -1):
        f[l] = max(f[l], f[l - w[i]] + v[i])
```
注：若正着遍历子背包，会影响后面的取值（可能出现一个物品被取多次的情况），所以必须反着更新
### 2.完全背包（每种物品数量不限）：
由于每种物品数量不限，开一个一维数组就可以
```python
n, a, b, c = map(int, input().split())  
dp = [0] + [-114514] * 114514  
for i in range(1, n + 1):  
    dp[i] = max(dp[i - a], dp[i - b], dp[i - c]) + 1  """相当于遍历"""
print(dp[n])
```
### 3.多重背包（每种物品数量有上限）：
基本想法是拆成0-1背包，但是大概率TLE，所以一般采用更高效的方法：改变拆法，一般是二进制优化（比如把14拆成1+2+4+7）,当然，涉及到二进制就肯定用到位运算了，这里实际上用到了状态压缩的思想
题：ls的1~n表示硬币面值，n+1~2n表示数量，m为金额上限，要求1~m能支付多少个数
```python
import math  
while True:  
    n, m = map(int, input().split())  
    if n == 0 and m == 0:  
        break  
    ls = list(map(int, input().split()))  
  
    w = (1 << (m + 1)) - 1
    """这里表示状态压缩，每一位分别对应0,1,2……m能否被表示，加上0是因为后续位运算需要一个1打头
    让一个0能被表示最简单"""
    result = 1  """表示0能被取到"""
  
    for i in range(n):  
        number = ls[i + n] + 1 
        limit = int(math.log(number, 2)) """进行二进制拆分"""
        rest = number - (1 << limit) """余下的部分"""
        
        for j in range(limit):  
            result = (result | (result << (ls[i] * (1 << j)))) & w  
            """或运算表示与之前的结果取交集（相当于动态规划），而更新硬币后，若原来i能被表示，
            现在i + 硬币面值能被表示，对应在result里就是它的后面值倍能被表示，也就是乘上2的这
            么多倍，也就是右移"""
        if rest > 0:  
            result = (result | (result << (ls[i] * rest))) & w  
  
    print(bin(result).count('1') - 1)
```
### 4.恰好型背包：
题：第一行总时间和组数，之后n行是时间和效果，要求时间恰好为t，求最好效果
注：难度一般，但很多细节需要注意
```python
t, n = map(int, input().split())  
time, weight = [0], [0]  
for _ in range(n):  
    a, b = map(int, input().split())  
    time.append(a)  
    weight.append(b)  
    
dp = [[-1] * (t + 1) for _ in range(n + 1)]  
"""注意第0行也要初始化，不然第一行可能会有错误（本题不会，第一行最开始对应最后一行）"""
for i in range(n + 1):  
    dp[i][0] = 0  
"""每一列第一个初始化为0，不然后续dp无法推进"""
for i in range(1, n + 1):  
    for j in range(1, t + 1):  
        dp[i][j] = dp[i - 1][j]  
        """代表不选的状态，若不做这个复制，该处的dp值还是
        -1，但实际上不一定是，没有有效更新"""
        if j - time[i] >= 0 and dp[i - 1][j - time[i]] >= 0:  
            dp[i][j] = max(dp[i][j], dp[i - 1][j - time[i]] + weight[i])  
print(dp[-1][-1])
```
滚动数组优化写法：
```python
t,n=map(int,input().split())
dp=[0]+[-1]*(t+1)
for i in range(n):
    k,w=map(int,input().split())
    for j in range(t,k-1,-1):
    """注意0-1背包反向遍历"""
        if dp[j-k]!=-1:
            dp[j]=max(dp[j-k]+w,dp[j])
print(dp[t])
```
## 4.公共子字符串问题：
### 1.不要求连续：
得到的就是一个不减的序列，每次出现相等的字符都会在上一个的基础上加1
```python
a, b = input().split()  
m, n = len(a), len(b)  
dp = [[0] * (n + 1) for _ in range(m + 1)]  
for i in range(1, m + 1):  
    for j in range(1, n + 1):  
        if a[i - 1] == b[j - 1]:  
            dp[i][j] = dp[i - 1][j - 1] + 1  
        else:  
            dp[i][j] = max(dp[i][j - 1], dp[i - 1][j])  
print(dp[-1][-1])
```
### 2.要求连续：
不相等后清0即可
```python
a, b = input().split()  
m, n = len(a), len(b)  
dp = [[0] * (n + 1) for _ in range(m + 1)]  
res = 0  
for i in range(1, m + 1):  
    for j in range(1, n + 1):  
        if a[i - 1] == b[j - 1]:  
            dp[i][j] = dp[i - 1][j - 1] + 1  
        else:  
            dp[i][j] = 0  
        res = max(res, dp[i][j])  
print(res)
```
## 5.素数问题：
### 1.找素数（大量）：
算法名字叫线性欧拉筛法，核心思想就是动态规划打表，不再赘述，假设我们已经找到了上界m
```python
is_prime = [True] * (m + 1)
is_prime[0] = is_prime[1] = False
primes = []
for i in range(2, m + 1):
	if is_prime[i]:
		prime.append(i)
	for p in primes: """每一次得到一个i我们都把素数表中这些素数的倍数标记，它们是合数"""
		if i * p > m:
			break
		is_prime[i * p] = False
		if i % p == 0:
			break
		"""这里是为了防止重复计算，之后的数可被更大的i标记"""
```
### 2.找最小素因子（大量）：
原理跟上面完全一样，仅仅修改几行代码即可
- is_prime初始化改为spf全为0
- 遍历2~m时的第一步判断改为判断是否为0
- 赋值改为spf\[i \* p] = p
- 最下面的判断改为p == spf\[i]
### 3.分解质因数：
```python
def pFactors(n):
	pFact, limit, num = [], int(n ** 0.5) + 1, n
	for p in range(2, limit):
		while num % p == 0:
			pFact.append(p)
			num //= p
	if num > 1:
		pFact.append(num)
	return pFact
```
# Dijkstra
## 1.用途：
处理有向无环加权图的最短路径问题
## 2.注意：
不能处理带负权的图，这种情况需要用贝尔曼福德算法
## 3.过程：
主要分四步：
- 找出当前最近节点（与起点相比），注意节点不会被重复操作，一旦操作过那么所得的就是最短路程及其路径
- 计算经过当前节点前往各个邻节点的权重
- 重复前两步
- 得出结果
总体为贪心和懒更新的思想
一个原则是，一个节点只要加工过，就达到全局最小，不能再加工，反之若没达到全局最小就说明没加工过，可以用这个来判断是否加工过，就不用再单开一个集合
## 4.优化：
由于中间需要找出最近节点，这一步的查找可以用最小堆来简化，查找的时间为O(1)，由于每一步加入元素都是用堆的操作实现，实际的时间复杂度也远没有O(n)
## 5.代码：
以一个矩阵代表节点的题目为例，如果是一般情况，相当于知道各边的权重，没有什么本质区别
```python
from heapq import heappush, heappop

def dijkstra(m, n, start):
	pq = []
	"""这是一个最小堆，第一个元素表示更新到现在为止从起点到该节点的花费，第二个表示节点"""
	heappush(pq, start)
	w = float('inf')
	cost = [[w] * n for _ in range(m)]
	"""这个列表用于记录当前从起点到该节点的花费，初始化为正无穷，采用懒更新来维护"""
	while pq:
		c, node = heappop(pq)
		x, y = node
		if c > cost[x][y]:
			continue
		"""如果从堆中弹出的花费比cost记录的花费要高，说明该节点已经加工过"""
		cost[x][y] = c
		for nx, ny in {邻居元组构成的集合}："""注意结合题目具体情况添加限制条件"""
			new_cost = c + weight """表示从(x, y)到(nx, ny)的花费"""
			if new_cost < cost[nx][ny]:
				cost[nx][ny] = new_cost
				heappush(pq, (new_cost, (nx, ny))
	return pq
```
# 区间
## 1.区间合并：
### 1.步骤：
- 按照区间左端点排序
- 维护前面区间中最右边的端点为ed。从前往后枚举每一个区间，判断是否应该将当前区间视为新区间。
### 2.代码实例：
```python
intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]  
intervals.sort(key=lambda x: x[0])  
res = []  
s, e = intervals[0][0], intervals[0][1]  
for i in intervals:  

    if i[0] > e:  
        res.append([s, e])  
        s, e = i[0], i[1]  
    else:  
        e = max(i[1], e)
        
res.append([s, e])  
print(res)
```
## 2.选择不相交区间：
### 1.原则：
**优先保留结束早的区间**，这样能为后续留下更多空间，从而保留更多区间。
### 2.步骤：
- 按照区间**右端点**从小到大排序。
- 从前往后依次枚举每个区间。
### 3.代码实例：
```python
intervals = [[0,2],[1,3],[2,4],[3,5],[4,6]]  
intervals.sort(key=lambda x: x[1])  
res = 0  
e = intervals[0][1]  
for i in intervals[1:]:  

    if i[0] < e:  
        res += 1  
        e = min(i[1], e)  
    else:  
        e = i[1]  
        
print(res)
```
## 3.区间选点：
### 1.大意：
给出一堆区间，取**尽量少**的点，使得每个区间内**至少有一个点**（不同区间内含的点可以是同一个，位于区间端点上的点也算作区间内）。
### 2.解法：
同2，对于这些**最大的不相交区间**，肯定是每个区间都需要选出一个点。而其他的区间都是和这些选出的区间有重复的，我们只需要把点的位置选在**重合**的部分即可
## 4.区间覆盖：
### 1.大意：
给出一堆区间和一个目标区间，问最少选择多少区间可以**覆盖**掉题中给出的这段目标区间。
### 2.步骤：
- 按照区间左端点从小到大排序。
- **从前往后**依次枚举每个区间，在所有能覆盖当前目标区间起始位置start的区间之中，选择**右端点**最大的区间。
- 更新start和end值即可
### 3.优化：
我们预处理所有的子区间，对于每一个位置 i，我们记录以其为左端点的子区间中最远的右端点，记为 maxn[i]
具体地，我们枚举每一个位置，假设当枚举到位置 i 时，记左端点不大于 i 的所有子区间的最远右端点为 last。这样 last 就代表了当前能覆盖到的最远的右端点。
每次我们枚举到一个新位置，我们都用 maxn[i] 来更新 last。如果更新后 last =i，那么说明下一个位置无法被覆盖，我们无法完成目标。
同时我们还需要记录上一个被使用的子区间的结束位置为 pre，每次我们越过一个被使用的子区间，就说明我们要启用一个新子区间，这个新子区间的结束位置即为当前的 last。也就是说，每次我们遇到 i= pre，则说明我们用完了一个被使用的子区间。这种情况下我们让答案加 1，并更新 pre 即可。
### 3.代码示例：
```python
clips = [[0,2],[4,6],[8,10],[1,9],[1,5],[5,9]]  
time = 10  
maxn = [0] * time  
last = res = pre = 0  
for a, b in clips:  
    if a < time:  
        maxn[a] = max(maxn[a], b)  
        
for i in range(time):  
    last = max(last, maxn[i])  
    if i == last:  
        print(-1)  
        exit()  
    if i == pre:  
        res += 1  
        pre = last  
        
print(res)
```
## 5.区间分组：
### 1.大意：
**区间分组**问题大概题意就是：给出一堆区间，问最少可以将这些区间分成多少组使得每个组内的区间互不相交。
### 2.步骤：
- 按照区间左端点从小到大排序。
- 从**前往后**依次枚举每个区间，判断当前区间能否被放到某个现有组里面。（即判断是否存在某个组的右端点在当前区间之中。如果可以，则不能放到这一组）
## 3.优化：
为了能快速的找到能够接收当前区间的组，我们可以使用**优先队列 （小顶堆）**。
优先队列里面记录每个组的右端点值，每次可以在 O(1) 的时间拿到右端点中的的最小值。
这个最小值要是不能放就都不能放了
### 4.代码实例：
```python
import heapq  
n = 2  
startEnd = [[1, 2], [2, 3]]  
startEnd.sort()  
q = []  

for i in range(n):  
    if not q or q[0] > startEnd[i][0]:  
        heapq.heappush(q, startEnd[i][1])  
    else:  
        heapq.heappop(q)  
        heapq.heappush(q, startEnd[i][1])  
        
print(len(q))
```
## 6.覆盖连续区间：
### 1.原理：
维护一个当前能表示的最大连续值，再扩展这个范围
### 2.代码实例：
情景：一些硬币覆盖1~X
```python
X, N = map(int, input().split())
coins = list(map(int, input().split()))
coins.sort(reverse=True)
r = 0
cn = 0
for i in range(1, X + 1):
    if i > cn:
        for t in coins:
            if t <= i:
                r += 1
                cn += t
                break
        else:
            r = - 1
            break
print(r)
```
# 不定行输入及数据缓存
## 1.异常捕获（方便debug）：
```python
while True:
	try:
		input()
	except EOFError:
		break
```
## 2.缓存读取：
### 1.逐行读取：
```python
from sys import stdin
for line in stdin:
	line = line.strip()
```
### 2.一次性读取：
```python
from sys import stdin
data = stdin.read()
lines = data.splitlines()
for line in lines:
	pass
```
### 3.与iter配合：
```python
from sys import stdin
data = iter(stdin.read().split())
```
# 递归回溯相关
## 1.全排列（稳定）：
### 1.代码：
```python
def permute(ls: list) -> list:  
    res = []  
    if len(ls) < 2:  
        return [ls]  
    for i in range(len(ls)):  
        for t in permute(ls[:i] + ls[i + 1:]):  
            res.append([ls[i]] + t)  
    return res
```
### 2.标准库：
itertools中的permutations
传入两个参数：
- iterable：需要进行排列的可迭代对象
- r：指定排列长度，不指定则默认全排列
返回一个迭代器，其中是每个排列对应的元组
## 2.性能优化：
### 1.functools.lru_cache:
可以缓存函数返回值，避免重复计算子问题
```python
@lru_cache(maxsize=None)
def recursive_function(n):
```
- **内存使用**：虽然 `lru_cache` 可以显著提高性能，但需要注意它会占用额外的内存来存储缓存结果。对于非常大的输入，可能会导致内存不足。
### 2.修改递归深度：
```python
from sys import setrecursionlimit(n)
```
### 3.yield生成器：
`yield` 是 Python 中用于定义生成器函数的关键字。生成器是一种特殊的迭代器，它允许你在函数内部逐步生成值，而不是一次性生成所有值并将它们存储在内存中。当你在函数中使用 `yield` 语句时，这个函数就变成了一个生成器。当调用生成器函数时，它不会立即执行函数体内的代码，而是返回一个生成器对象。只有当这个生成器对象被迭代时，才会执行函数体内的代码，直到遇到 `yield` 语句，此时函数会暂停执行，并返回 `yield` 后面的表达式的值。当再次迭代生成器时，函数会从上次暂停的地方继续执行，直到遇到下一个 `yield` 语句，依此类推，直到函数执行完毕。

**`yield` 与 `return` 的区别**
- **执行时机**：当函数中使用 `return` 时，函数会立即终止执行，并返回一个值；而使用 `yield` 时，函数会生成一个生成器对象，该对象可以在需要时逐步产生值。
- **内存占用**：`return` 需要一次性计算并返回所有的值，如果这些值的数量很大，可能会消耗大量的内存。相比之下，`yield` 可以按需生成值，因此更加节省内存。
- **可迭代性**：使用 `return` 的函数只能返回一次值，而使用 `yield` 的生成器可以多次产生值，使得生成器可以用于迭代。
- **状态保持**：`yield` 使函数能够记住其上一次的状态，包括局部变量和执行的位置，因此当生成器再次被调用时，它可以从中断的地方继续执行。而 `return` 则不会保存任何状态信息，每次调用都是全新的开始。

**使用 `yield` 的好处**
- **节省资源**：由于生成器是惰性求值的，只有在需要的时候才计算下一个值，所以它可以有效地处理大数据集，避免一次性加载所有数据到内存中。
- **简化代码**：生成器提供了一种简单的方式来实现复杂的迭代模式，而不需要显式地管理迭代状态。
- **提高效率**：对于需要连续处理大量数据的应用场景，生成器可以避免不必要的内存分配和垃圾回收，从而提高程序的运行效率。
- **易于使用**：生成器可以像普通迭代器一样使用，可以很容易地集成到现有的代码中，如 for 循环等。

综上所述，`yield` 提供了一种强大的机制，用于处理那些需要逐步生成或处理大量数据的情况，同时保持代码的简洁性和高效性。

给出一个生成全排列的例子：
```python
n = int(input())
l = []
for i in range(1,n+1):
    l.append(f'{i}')

def arrange(l):
    if len(l) == 1:
        """
        当列表中只有一个元素时，使用yield关键字返回这个元素。这里使用了生成器，而不是直接返回（return）值，
        这意味着函数可以暂停执行并在需要时恢复，这对于处理大量数据或递归调用非常有用。
        """
        yield l[0]
    else:
        for i in range(len(l)):
            new_l = l[:i] + l[i+1:]
            for rest in arrange(new_l):
                yield l[i] + ' ' + rest

for ans in arrange(l):
    print(ans)
```
## 3.子集：
### 1.库函数：
```python
from itertools import combinations
res = []
for i in range(len(nums) + 1):
	for tmp in combinations(nums, i):
		res.append(tmp)
```
- combinations介绍：
- itertools.combinations(iterable, r)
- **iterable**：输入的可迭代对象（如列表、字符串、元组等）。
- **r**：组合的长度（必须是非负整数）。
### 2.迭代：
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = [[]]
        for i in nums:
            res = res + [[i] + num for num in res]
        return res
```
### 3.回溯：
- 版本一
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        n = len(nums)
        
        def helper(i, tmp):
            res.append(tmp)
            for j in range(i, n):
                helper(j + 1,tmp + [nums[j]] )
        helper(0, [])
        return res  
```
- 版本二
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans, sol = [], []
        
        def backtrack(i):
            """终止条件：处理完所有元素"""
            if i == n:
                ans.append(sol[:])
                return
            
            """分支1：不选择 nums[i]"""
            backtrack(i + 1)
            
            """分支2：选择 nums[i]"""
            sol.append(nums[i])
            backtrack(i + 1)
            sol.pop()  """回溯"""
        
        backtrack(0)
        return ans
```
# BFS
## 1.基本模板：
```python
from collections import deque
q = deque([(0, start)])
in_queuq = {start} """快速检索，避免重复入队"""
while q:
	step, pos = q.popleft()
	if pos == end:
		print(step)
	"""之后执行入队操作，记得更新相关量，若要找到最短路径，in_queue集合可用一个记录父节点的列     表"""
```
## 2.多源BFS：
直接上例子，核心就是许多东西一次性入队
给出一个n\*n矩阵，0表示海洋，1表示陆地，距离用曼哈顿距离计算，请你找出一个海洋单元格，这个海洋单元格到离它最近的陆地单元格的距离是最大的，并返回该距离。
核心代码：
```python
from collections import deque
q = deque([])
d = [(0, 1), (0, -1), (1, 0), (-1, 0)]
for i in range(n):
	for j in range(n):
		if grid[i][j] == 1:
		q.append((i, j))
res = 0

while q:
	for _ in range(len(q)):
	"""这里的目的是逐层操作，每层的操作次数就是当前q的长度，操作了几层最终答案就是对应的数"""
		x, y = q.popleft()
		for dx, dy in d:
			nx, ny = x + dx, y + dy
			if 0 <= nx < n and 0 <= ny < n and grid[nx][ny] == 0:
				q.append((nx, ny))
				grid[nx][ny] = 1 """相当于标记"""
	res += 1 """一层操作完"""
```
