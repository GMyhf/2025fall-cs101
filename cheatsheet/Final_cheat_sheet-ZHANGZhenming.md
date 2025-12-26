# **cs101 Algo-DS CHEAT-SHEET**
## **——A C++ ver.**
Inspired by 汤伟杰, Translated & Optimized by 张真铭

### I.DP
#### 1.背包问题
初始化：如果要求恰好装满则为-inf；如果有可能装不满为0
1. **01背包** --> 每个物品只能拿一次
```cpp
//小偷背包 V-总容量,n-物品个数,cost=[0,     ],price=[0,     ]
void zero_one_bag(int V, int n, const vector<int>& cost, const vector<int>& price) {
    vector<int> dp(V + 1, 0);
    for (int i = 1; i <= n; i++)           //每个物品
        for (int j = V; j >= cost[i]; j--)    //逆向遍历每个容量
            dp[j] = max(dp[j], price[i] + dp[j - cost[i]]);
    return dp[V];
}
```
2. **完全背包** --> 每个物品可以拿无限次
```cpp
//零钱找零
void total_bag(int V, int n, const vector<int>& cost, const vector<int>& price) {
    vector<int> dp(V + 1, 0);
    for (int i = 1; i <= n; i++)           //每个物品
        for (int j = cost[i]; j <= V; j++)    //正向遍历每个容量
            dp[j] = max(dp[j], price[i] + dp[j - cost[i]]);
    return dp[V];
}
```
3. **多重背包** --> 每个物品的个数有限制,把每个物品的个数拆成1 2 4等转化为01背包
```cpp
//NBA门票
int many_bag(int V, int n, const vector<int>& cost, const vector<int>& price, vector<int>& s) {
    vector<int> dp(V + 1, 0);
    for (int i = 1; i <= n; ++i) {
        int k = 1;
        while (s[i] > 0) {
            int cnt = min(k, s[i]); // 转化为 01 背包处理
            for (int j = V; j >= cnt * cost[i]; --j)
                dp[j] = max(dp[j], dp[j - cnt * cost[i]] + cnt * price[i]);
            s[i] -= cnt;
            k *= 2;
        }
    }
    return dp[V];
}
```
4. **二维费用背包**
```cpp
int two_dimension_cost(int n,int V1,int V2,const vector<int> &cost1,const vector<int> &cost2,const vector<int> &price) {
    vector<vector<int>> dp(V1 + 1, vector<int>(V2 + 1, 0));
    for (int i = 0; i < n; i++)
        for (int c1 = V1; c1 >= cost1[i]; c1--)
            for (int c2 = V2; c2 >= cost2[i]; c2--)
                dp[c1][c2] = max(dp[c1][c2], dp[c1 - cost1[i]][c2 - cost2[i]] + price[i]);
    return dp[V1][V2];
}
```
---
#### 2.整数分割问题
1. 把n划分为若干个正整数，**不考虑顺序** --> 完全背包
4：4=3+1=2+2=2+1+1=1+1+1+1 共5种
```cpp
int divide1(int n) {
    vector<int> dp(n + 1, 0);
    dp[0] = 1;    //把0划分只有0这一种
    for (int i = 1; i <= n; i++)           //每个数字
        for (int j = i; j <= n; j++)            //正向遍历每个容量（每个n）
            dp[j] += dp[j - i];
    return dp[n];
}
```
2. 把n划分为若干个正整数，**考虑顺序**
4：4=3+1=1+3=2+2=2+1+1=1+2+1=1+1+2=1+1+1+1 共8种
```cpp
int divide2(int n) {
    vector<int> dp(n + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= n; i++)           //每个容量（每个n）
        for (int j = 1; j <= i; j++)            //每个可能划分出的数字
            dp[i] += dp[i - j];
    return dp[-1];
}
```
3. 把n划分为若干个不同的正整数，不考虑顺序 --> 01背包
4：4=3+1 共1种
```cpp
int divide3(n) {
    vector<int> dp(n + 1, 0);
    dp[0] = 1;
    for (int i = 1; i <= n; i++)
        for (int j = n; j >= i; j--)
            dp[j] += dp[j - i];
    return dp[n];
}
```
4. 把n划分为k个正整数，不考虑顺序
```cpp
//放苹果
int divide4(int n,int k) {
    vector<vector<int>> dp(n + 1, vector<int>(k + 1, 0));//dp[n][k]:把n分成k组
    for (int i = 1; i <= n; i++)
        dp[i][1] = 1;//每个数字分成1组都是1种
    for (int i = 2; i <= n; i++)
        for (int j = 2; j <= k; j++)
            if (i >= j) dp[i][j] = dp[i - 1][j - 1] + dp[i - j][j];
    return dp[n][k]; //dp[-1][-1]
}
```
---
#### 3.序列dp+数学归纳思维
已知dp[0]到dp[i-1]的所有状态，求出dp[i]，即找出dp[i]与之前状态的关系。
常见定义：dp[i]:到第i个位置时的状态（最大值等）,
dp[i,j]:从第i个位置到第j个位置时的状态，或到第i个位置时恰好为状态j。
例题：
| 题目              | 链接                                                                    | 递推式                                                            |
| ----------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------- |
| oj-核电站-09267   | http://cs101.openjudge.cn/practice/09267                                | `dp[i]=2*dp[i-1]-dp[i-1-m]`                                       |
| cf-Flowers-474D   | https://codeforces.com/problemset/problem/474/D                         | `dp[i]=dp[i-1]+dp[i-k]`                                           |
| lc-最长回文子串-5 | https://leetcode.cn/problems/longest-palindromic-substring/description/ | `dp[i,j]=dp[i+1,j-1] if s[i]==s[j] else False`                    |
| lg-合并石头-P1775 | https://www.luogu.com.cn/problem/P1775                                  | `dp[i,j]=min(dp[i,j],dp[i,k]+dp[k+1,j]+cost) for k in range(i,j)` |
```cpp
//合并石子 (时间复杂度很高)
int mergeStone(int n,const vector<int> &s) {//初始化，求最小值
    vector<vector<int>> dp(n + 1, vector<int>(n + 1, 1e9)); //dp[i][j]为区间i - j合并为一堆的最小值
    for (int i = 1; i <= n; i++)
        dp[i][i] = 0;//初始化，dp[i][i]=0
    vector<int> prefix(n + 1, 0);
    for (int i = 1; i <= n; i++)
        prefix[i] = prefix[i - 1] + s[i - 1];//计算前缀和
    for (int L = 2; L <= n; L++)//dp
        for (int i = 1; i <= n - L + 1; i++) {
            int j = i + L - 1;
            for (int k = i; k < j; k++)
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k + 1][j] + prefix[j] - prefix[i - 1]);
        }
    return dp[1][n];
}
```
---
#### 4.设置多个dp数组+数学归纳思维
设置dp1和dp2两个数组记录两种状态，一般定义dp1[i]为取s[i]，dp2[i]为不取s[i]，再利用数学归纳思维找出转移方程
例题：
| 题目                         | 链接                                             | 递推式                                                                                                                       |
| ---------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| cf-Basketball Exercise-1195C | https://codeforces.com/problemset/problem/1195/C | `dp1[i]=max(dp1[i-1],h1[i]+dp2[i-1]);dp2[i]=max(dp2[i-1],h2[i]+dp1[i-1])`                                                    |
| oj-红蓝玫瑰-25573            | http://cs101.openjudge.cn/practice/25573         | `dpr[i]=dpr[i-1] if s[i]==red else min(dpr[i-1]+1,dpb[i-1]+1);dpb[i]=dpb[i-1] if s[i]==blue else min(dpb[i-1]+1,dpr[i-1]+1)` |

---
#### 5.Kadane算法
oj-最大子矩阵-02766 http://cs101.openjudge.cn/practice/02766/
>Kadane算法
>一种非常高效的算法，用于求解一维数组中 最大子数组和。它能够在 O(n) 时间复杂度内解决问题，广泛应用于许多动态规划问题中。
>避免了计算前缀和数组
```cpp
int kadane(const vector<int> &s) { //一维
    int curr_max = s[0], total_max = s[0];
    for (int i = 1; i < s.size(); i++) {
        curr_max = max(curr_max + s[i], s[i]);
        total_max = max(total_max, curr_max);
    }
    return total_max;
}
```
```cpp
int kadane(const vector<int> &s) { //二维，压缩到一维数组
    int curr_max = s[0], total_max = s[0];
    for (int i = 1; i < s.size(); i++) {
        curr_max = max(curr_max + s[i], s[i]);
        total_max = max(total_max, curr_max);
    }
    return total_max;
}
int max_sum_matrix(const vector<int> &s) { //上下压缩
    int max_sum = -INT_MAX;
    int row = mat.size(), col = mat[0].size();
    for (int top = 0; top < row; top++) {
        vector<int> col_sum(col, 0);
        for (int bottom = top; bottom < row; bottom++) {
            for (int c = 0; c <= col; c++)
                col_sum[c] += s[bottom][c];
            max_sum = max(max_sum, kadane(col_sum));
        }
    }
    return max_sum;
}
```
---
#### 6.二维dp
| 题目                                     | 链接                                                    | 递推式                                                                                                                                                                                                                                                                                                       |
| ---------------------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| LC 72 双序列比对                         | https://leetcode.cn/problems/edit-distance/description/ | `if (word1[i - 1] == word2[j - 1])dp[i][j] = dp[i - 1][j - 1];else dp[i][j] = min(dp[i - 1][j - 1], min(dp[i][j - 1], dp[i - 1][j])) + 1;`                                                                                                                                                                   |
| CF 698A 不能选择连续的两个同一性质的序列 | https://codeforces.com/problemset/problem/698/A         | `dp[i][0] = min(dp[i - 1][0], min(dp[i - 1][1], dp[i - 1][2])) + 1;if (a[i] == 1)dp[i][1] = min(dp[i - 1][2], dp[i - 1][0]);else if (a[i] == 2) dp[i][2] = min(dp[i - 1][1], dp[i - 1][0]); else if (a[i] == 3) { dp[i][1] = min(dp[i - 1][2], dp[i - 1][0]); dp[i][2] = min(dp[i - 1][1], dp[i - 1][0]); }` |
>同样的还有CF 455A，类似“打家劫舍”，不能同时选择两个相邻的项
```cpp
size_t n, m = 0;
cin >> n;
map<size_t, size_t> a;
for (size_t i = 0; i < n; i++) {
    size_t num;
    cin >> num;
    m = max(m, num);
    a[num]++;
}
vector<vector<size_t>> dp(m + 1, vector<size_t>(2, 0));
for (size_t i = 1; i <= m; i++) {
    dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
    dp[i][1] = dp[i - 1][0] + a[i] * i;
}
cout << max(dp[m][0], dp[m][1]) << '\n';
return 0;
```
### II.Dilworth Theory
>**最少单调链个数===最长反单调链长度**
>找最长上升子序列的长度，用lower_bound
>找最长下降子序列，先reverse，再用lower_bound
>如果是不降，用upper_bound
>如果是不升，先reverse，再用upper_bound
>看题目要求的最终结果是否需要相同元素的考虑，需要考虑用lower_bound，不需要用upper_bound
```cpp
int lengthOfLIS(const vector<int>& nums) {
    if (nums.empty()) return 0;
    vector<int> tails;
    for (int num : nums) {
        // lower_bound: 找到第一个 >= num 的位置
        auto it = lower_bound(tails.begin(), tails.end(), num);
        if (it == tails.end())
            tails.push_back(num);
        else
            *it = num;
    }
    return tails.size();
} // 若求不下降，改为 upper_bound
```
---
### III.PREFIX SUM
#### 1.前缀和数组
用于处理**多次查询**从[l,r]的序列之和的问题
```cpp
//二维前缀和
PrefixSum2D(const vector<vector<int>>& matrix) {
    m = matrix.size();
    n = matrix[0].size();
    preSum.assign(m + 1, vector<int>(n + 1, 0));
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            preSum[i][j] = preSum[i-1][j] + preSum[i][j-1] - preSum[i-1][j-1] + matrix[i-1][j-1];
}
int query(int x1, int y1, int x2, int y2) {// 查询子矩阵和
    return preSum[x2+1][y2+1] - preSum[x1][y2+1] - preSum[x2+1][y1] + preSum[x1][y1];
}
```
---
#### 2.前缀和的特殊用法(哈希表)
使用prefix和prefix_map来记录已有的前缀和，从而判断子串和为0的子串个数；或找相同前缀和数字出现的最远位置
例题：
| 题目                          | 链接                                             |
| ----------------------------- | ------------------------------------------------ |
| oj-完美的爱-27141             | http://cs101.openjudge.cn/practice/27141/        |
| cf-Kousuke's Assignment-2033D | https://codeforces.com/problemset/problem/2033/D |
```cpp
//cf-Kousuke's Assignment-2033D 找出不重叠的和为0的子序列个数，一旦找到就将prefixed集合清空
void solve_zero_sum_subarrays() {
    int n; cin >> n; vector<ll> a(n);
    for (int i = 0; i < n; ++i) cin >> a[i];
    ll prefix = 0; set<ll> prefixed = {0}; int cnt = 0;
    for (ll x : a) {
        prefix += x;
        if (prefixed.count(prefix))
            cnt++, prefix = 0, prefixed.clear(), prefixed.insert(0);
        else
            prefixed.insert(prefix);
    }
    cout << cnt << endl;
}
```
---
### IV.SORTING
#### 1.拓扑排序
```cpp
for(auto x: edge) {
        int fi=x.first, se=x.second;
        e[fi].push_back(se);     //邻接表存储
        pre[se]++;               //统计入度
    }
queue<int>q = queue<int>();
vector<int>order = vector<int>();
for(int i=1;i<=n;i++)
    if(pre[i]==0) q.push(i);
while(!q.empty()) {
    int x=q.front();
    q.pop(), order.push_back(x);
    for(auto y: e[x]) {
        pre[y]--;            //x已经被排序了，因此x的后继节点入度应该-1
        if(pre==0) q.push(y);
    }
}
```
#### 2.归并排序 --> 递归
```cpp
vector<int> merge(const vector<int>& left, const vector<int>& right) {
    vector<int> ans;
    int i = 0, j = 0;
    while (i < left.size() && j < right.size())
        if (left[i] < right[j])
            ans.push_back(left[i]), i++;
        else
            ans.push_back(right[j]), j++;// 比较两个数组的元素，依次将较小的元素放入结果数组
    while (i < left.size())
        ans.push_back(left[i]), i++;
    while (j < right.size())
        ans.push_back(right[j]), j++;// 将剩余的元素添加到结果数组
    return ans;
}
vector<int> merge_sort(const vector<int>& s) {
    if (s.size() <= 1)
        return s;// 递归终止条件：数组长度小于等于1
    int mid = s.size() / 2;// 计算中点
    vector<int> left(s.begin(), s.begin() + mid);// 分割左半部分
    vector<int> right(s.begin() + mid, s.end());// 分割右半部分
    left = merge_sort(left);
    right = merge_sort(right);// 递归排序左右部分
    return merge(left, right);// 合并排序后的左右部分
}
```
```cpp
//lc-LCR085-括号生成
vector<string> generateParenthesis(int n) {
    vector<string> ans;
    auto dfs = [&](this auto&& dfs, int i, int left, string path) {
        if (i == 2 * n) { ans.push_back(path); return; }
        if (left < n)
        dfs(i + 1, left + 1, path + '(');
        if (i - left < left)
            dfs(i + 1, left, path + ')');
    };
    dfs(0, 0, "");
    return ans;
}
```
---
### V.SEARCHING
#### 1.dfs
1. 无回溯操作
2. 有回溯操作
模板是：①有退出条件 ②递归之间做重复要做的事情 ③递归之后回溯为原状态

例题：
```cpp
//oj-八皇后-02754
/*考虑以下递归步骤：在某次递归时，curr = [1, 5, 8, 6]，此时 ans.append(curr)。接下来，回溯修改了 curr，变为 [1, 5, 8, 7]。由于 ans 中保存的是 curr 的引用，ans 中原本存储的 [1, 5, 8, 6] 也会变为 [1, 5, 8, 7]。因此使用 curr[:]，创建当前列表的拷贝，确保后续对 curr 的修改不会影响已保存的解*/
int ans_cnt = 0;
vector<int> curr;
bool col[9], diag1[17], diag2[17]; // 辅助数组加速判断
void dfs_queen(int row) {
    if (row == 8) {
        ans_cnt++; // 或保存 curr
        return;
    }
    for (int c = 0; c < 8; ++c) {// 检查列、主对角线(row+c)、副对角线(row-c+8)
        if (!col[c] && !diag1[row + c] && !diag2[row - c + 8]) {
            col[c] = diag1[row + c] = diag2[row - c + 8] = true;
            curr.push_back(c + 1);
            dfs_queen(row + 1);
            curr.pop_back();// 回溯
            col[c] = diag1[row + c] = diag2[row - c + 8] = false;
        }
    }
}
```
```cpp
//oj-有界的深度优先搜索-23558
int n, m, l, st;
vector<int> g[100];
vector<int> ans;
void dfs(vector<bool> &vis, int st, int depth) {
    if (depth == l) return;
    for (const auto &u : g[st])
        if (!vis[u])
            vis[u] = true,ans.push_back(u),dfs(vis, u, depth + 1);
}
```
---
#### 2.BFS
逐层扩展，用来求最小步数，模板；如果想保留路径，可以把路径作为参数传递，其中双端队列q加入的元素可能是三维，包含坐标和时间或者步数或者路径等等。
```cpp
struct Point { int x, y, step; };
int bfs(int sx, int sy, int ex, int ey, int n, int m, const vector<string>& grid) {
    queue<Point> q;
    q.push({sx, sy, 0});
    vector<vector<bool>> visited(n, vector<bool>(m, false));
    visited[sx][sy] = true;
    int dx[] = {0, 0, 1, -1};
    int dy[] = {1, -1, 0, 0};
    while (!q.empty()) {
        Point p = q.front(); q.pop();
        if (p.x == ex && p.y == ey) return p.step;
        for (int i = 0; i < 4; ++i) {
            int nx = p.x + dx[i], ny = p.y + dy[i];
            if (nx >= 0 && nx < n && ny >= 0 && ny < m && grid[nx][ny] != '#' && !visited[nx][ny]) {
                visited[nx][ny] = true;
                q.push({nx, ny, p.step + 1});
            }
        }
    }
    return -1;
}
```
---
#### 3.Dijkstra算法
解决单源最短路径问题，用于非负权图，使用`heapq`的最小堆来代替`bfs`中的`deque`，设置`dist`列表更新最短距离。
例题：
```cpp
struct Node {
    int u, dist;
    bool operator>(const Node& other) const { return dist > other.dist; }
};
int dijkstra(int start, int end, int n, const vector<vector<pair<int, int>>>& adj) {
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    vector<int> dist(n + 1, INF);
    dist[start] = 0;
    pq.push({start, 0});
    while (!pq.empty()) {
        int u = pq.top().u;
        int d = pq.top().dist;
        pq.pop();
        if (d > dist[u]) continue; // 懒删除机制
        if (u == end) return d;
        for (auto& edge : adj[u]) {
            int v = edge.first;
            int w = edge.second;
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({v, dist[v]});
            }
        }
    }
    return -1;
}
```
---
### VI.DATA STRUCTURE
#### 1.STACK
栈(stack)，使用`list`来模拟，遵循后进先出的原则。
例题：
```cpp
//oj-快速堆猪-22067
//辅助栈
stack<int> st, min_st; string s;
while (cin >> s) {
    if (s == "push") {
        int n; cin >> n;
        st.push(n);
        if (min_st.empty() || n <= min_st.top()) min_st.push(n);
    }
    else if (s == "pop") {
        if (st.top() == min_st.top()) min_st.pop();
        st.pop();
    }
    else if (s == "min") cout << min_st.top() << endl;
}
```
---
##### 卡特兰数
Catalan数$𝐶_𝑛$的递推关系有着天然的递归结构：规模为$𝑛$的计数问题$𝐶_𝑛$，可以通过枚举分界点，分拆为两个规模分别为$𝑖$和$(𝑛−1-𝑖)$的子问题。这一递推关系使得Catalan数广泛出现于各类具有类似递归结构的问题中。
**路径计数问题**：有一个大小为
$n\times n$ 的方格图，左下角为 $(0,0)
(0, 0)$，右上角为 $(𝑛,𝑛)
(n, n)$。从左下角开始，每次都只能向右或者向上走一单位，不走到对角线
$y=x$ 上方（但可以触碰）的情况下，到达右上角的路径总数为 $
C_n$。
**圆内不相交弦计数问题**：圆上有
$2n$ 个点，将这些点成对连接起来且使得所得到的 
$n$ 条线段两两不交的方案数是
$C_n$。
**三角剖分计数问题**：对角线不相交的情况下，将一个凸 
$(n+2)$ 边形区域分成三角形区域的方法数为
$C_n$。
**二叉树计数问题**：含有
$n$ 个结点的形态不同的二叉树数目为 
$C_n$。等价地，含有
$n$ 个非叶结点的形态不同的满二叉树数目为
$C_{n}$。
**括号序列计数问题**：由
$n$ 对括号构成的合法括号序列数为
$C_n$。
**出栈序列计数问题**：一个栈（无穷大）的进栈序列为
$1,2,3, \ldots ,n$，合法出栈序列的数目为
$C_n$。
**数列计数问题**：由
$n$ 个 
$+1$ 和 
$n$ 个
$-1$ 组成的数列 $a_1,a_2, \ldots ,a_{2n}$ 中，部分和满足$a_1+a_2+ \ldots +a_k \geq 0~(k=1,2,3, \ldots ,2n)$ 的数列数目为
$C_n$。

$C_n = \frac{(4n - 2)}{n + 1} C_{n-1},\ n > 0,\ C_0 = 1.$


---
此外，还有常用的**单调栈**(monotonic stack)，其优点是：若维护了一个单调递增栈，
则每次取出栈顶元素时，**新的栈顶元素**和**不符合条件而未入栈的元素**恰好是**取出的元素两侧的距离最近的比其小的元素**；单调递减栈类似。
例题：
```cpp
//lc-接雨水-42 维护递减栈。单调栈的好处是：由于维护的是单调下降的高度，当弹出栈顶元素的，其左侧就是左侧第一个比它高的元素而弹出操作也意味着右侧就是右侧第一个比它高的元素
int trap(vector<int>& height) {
    stack<int> st; // 存下标
    int ans = 0;
    for (int i = 0; i < height.size(); ++i) {
        while (!st.empty() && height[st.top()] < height[i]) {
            int top = st.top(), st.pop();
            if (st.empty()) break;
            int w = i - st.top() - 1, h = min(height[i], height[st.top()]) - height[top];
            ans += w * h;
        }
        st.push(i);
    }
    return ans;
}

```
```cpp
//oj-护林员盖房子-21577
int house(mat,n,m) { //预处理，逐层计算，转换为求最大矩形面积(lc-最大矩阵-85)
    vector<int> height(m + 1, 0);
    int ans = 0;
    for (const auto &i : arr) {
        for (int j = 0; j < m; j++)
            if (i[j] == 0) height[j]++;
            else height[j] = 0;
        stack<int> st;
        st.push(-1);
        for (int j = 0; j <= m; j++) {
            while (st.top() != -1 && height[st.top()] > height[j])
                int top = st.top(), st.pop(), ans = max(ans, height[top] * (j - st.top() - 1));
            st.push(j);
        }
    }
    return ans;
}
```
---
#### 2.HEAPQ
**最小堆**(heapq)可以维护列表中的最小值并将其位置放在第一个，即heap[0]。如果想得到最大值，以负值形式存入。
且最小堆通常涉及到内部元素的删除，而内置函数无此操作，则会利用到**懒删除**操作，使用字典记录已被删除的元素，需要取最小值时再一次性删除。
例题：
```cpp
//懒删除 oj-快速堆猪-22067
stack<int> st;
priority_queue<int, vector<int>, greater<int>> heap;
unordered_map<int, int> out;
string s;
while (cin >> s) {
    if (s == "pop") {
        if (!st.empty())
            out[st.top()]++, st.pop();
    } else if (s == "min") {
        while (!heap.empty() && out[heap.top()] > 0)
            out[heap.top()]--, heap.pop();
        if (!heap.empty()) cout << heap.top() << endl;
    } else if (s == "push") {
        int n; cin >> n;
        st.push(n), heap.push(n);
    }
}
```
```cpp
//oj-剪绳子-18164
long long min_Heap(long long arr[], long long n) {
    priority_queue<long long, vector<long long>, greater<long long>> minHeap;
    for (int i = 0; i < n; i++)
        minHeap.push(arr[i]);
    long long totalCost = 0;
    while (minHeap.size() > 1) {
        long long first = minHeap.top();
        minHeap.pop();
        long long second = minHeap.top();
        minHeap.pop();
        totalCost += first + second;
        minHeap.push(first + second);
    }
    return totalCost;
}
```
**后悔解法** cf-potions-1526C1 tags:data structure,greedy
```cpp
void solve_potions() {
    int n; cin >> n;
    priority_queue<int, vector<int>, greater<int>> pq; // 小顶堆存负数部分
    ll health = 0; int count = 0;
    for (int i = 0; i < n; ++i) {
        int x; cin >> x;
        health += x, pq.push(x), count++;
        while (health < 0)
            health -= pq.top(), pq.pop(), count--;
    }
    cout << count << endl;
}


```
---
### VII.INTERVAL PROBLEMS
区间合并问题常常涉及到对区间左端点或者右端点的排序。
eg：
1. 合并所有有交集的区间，返回最终个数--对左端点排序，不断更新右边界
```cpp
//s=[(l1,r1),(l2,r2),...,(ln,rn)]
vector<pair<int, int>> merge_intervals(vector<pair<int, int>>& intervals) {
    if (intervals.empty()) return {};
    sort(intervals.begin(), intervals.end()); // 默认按 first 升序
    vector<pair<int, int>> res;
    res.push_back(intervals[0]);
    for (size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= res.back().second) {
            res.back().second = max(res.back().second, intervals[i].second);
        } else
            res.push_back(intervals[i]);
    }
    return res;
}
```
2. 选择尽量多的无交集的区间，返回最大数量--对右端点排序
```cpp
// 类似于活动安排问题
int max_non_overlapping(vector<pair<int, int>>& intervals) {
    sort(intervals.begin(), intervals.end(), [](auto& a, auto& b){ return a.second < b.second; });// 按结束时间排序
    int count = 0;
    int end = -INF;
    for (auto& p : intervals)
        if (p.first >= end) // 注意题目是 > 还是 >=
            count++, end = p.second;
    return count;
}
```
3. oj-进程检测-04100(区间选点)--对右端点排序
```cpp
sort(t.begin(), t.end(), [](const pair<int, int> &a, const pair<int, int> &b) { return a.second < b.second; });
int curr = 0, cnt = 0;
for (const auto &i : t)
    if (i.first > curr)
        curr = i.second, cnt++;
```
4. 区间覆盖--对左端点排序，从起点开始每次选最远的右端点
```cpp
//oj-世界杯只因-27104
int minCam(const vector<int> &ranges) {
    int n = ranges.size(), m = *max_element(ranges.begin(), ranges.end());
    int curr = 0, next = 0, cnt = 0;
    while (curr < n) {
        next = curr + ranges[curr] + 1;
        for (int i = max(0, curr - m); i < min(n, curr + m + 1); i++)
            if (i - ranges[i] <= curr)
                next = max(next, i + ranges[i] + 1);
        cnt++, curr = next;
    }
    return cnt;
}
```
5. 主持人调度--对左端点排序-转为事件（在排序时增加第二个元素, 差分/扫描线）
```cpp
int min_hosts(const vector<pair<int, int>>& intervals) {
    vector<pair<int, int>> events;
    for (auto& p : intervals) {
        events.push_back({p.first, 1});  // 开始，+1
        events.push_back({p.second, -1}); // 结束，-1
    }
    sort(events.begin(), events.end()); // 按时间排序，时间相同先处理哪个视开闭区间而定
    int curr = 0, max_needed = 0;
    for (auto& e : events)
        curr += e.second, max_needed = max(max_needed, curr);
    return max_needed;
}
```
类似的，将区间转换为事件，遍历事件的两个端点的例题：
```cpp
// cf-Best Price-2051E
vector<pair<size_t, size_t>> events;
size_t ipt;
for (int i = 0; i < n; i++) {
    cin >> ipt; events.push_back({ipt, 1});
}
for (int i = 0; i < n; i++) {
    cin >> ipt; events.push_back({ipt, 2});
}
sort(events.begin(), events.end());
int i = 0, bad = 0;
size_t cost = 0, p = n;
while (i < 2 * n) {
    int curr = events[i].first;
    if (bad <= k) cost = max(cost, p * events[i].first);           
    while (i < 2 * n && events[i].first == curr)
        bad += (events[i].second == 1), bad -= (events[i].second == 2), p -= (events[i].second == 2), i++;
}
cout << cost << '\n';
```

### VIII.Disjoint, Trie, Manacher & Sliding Window
#### 1.DSU
例题：
```cpp
// oj 01182 食物链
vector<int> pa, sz;
void _init(int n) {//按题意修改
    pa.resize(3 * n);
    for (int i = 0; i < 3 * n; i++) pa[i] = i;
    sz.assign(3 * n, 1);
}
int _find(int i) {
    return pa[i] == i ? pa[i] : pa[i] = _find(pa[i]);
}
void _union(int i, int j) {
    int irep = _find(pa[i]), jrep = _find(pa[j]);
    if (irep == jrep) return;
    if (sz[irep] < sz[jrep]) pa[irep] = jrep, sz[jrep] += sz[irep];
    else pa[jrep] = irep, sz[irep] += sz[jrep];
}
int main() {
    int N, K; cin >> N >> K; _init(N); int ans = 0;
    while (K--) {
        int D, X, Y; cin >> D >> X >> Y;
        if (X > N || Y > N) ans++, continue;
        if (D == 1) {
            if (_find(X + N - 1) == _find(Y - 1) || _find(X - 1) == _find(Y + N - 1)) ans++, continue;
            _union(X - 1, Y - 1); _union(X + N - 1, Y + N - 1); _union(X + 2 * N - 1, Y + 2 * N - 1);
        }
        else {
            if (_find(X - 1) == _find(Y - 1) || _find(Y + N - 1) == _find(X - 1)) ans++, continue;
            _union(X + N - 1, Y - 1); _union(Y + 2 * N - 1, X - 1); _union(X + 2 * N - 1, Y + N - 1);
        }
    }
    cout << ans << '\n'; return 0;
}
```
### 2. Trie
```cpp
class Trie {
private:
    vector<Trie*> children;
    bool isLeaf;
    Trie* searchPrefix(string prefix) {
        Trie* node = this;
        for (auto c : prefix) {
            if (node->children[c - 'a'] == nullptr)
                return nullptr;
            node = node->children[c - 'a'];
        }
        return node;
    }
public:
    Trie() : children(26), isLeaf(false) {}
    auto insert(string word) {
        Trie* node = this;
        for (auto c : word) {
            if (node->children[c - 'a'] == nullptr)
                node->children[c - 'a'] = new Trie();
            node = node->children[c - 'a'];
        }
        node->isLeaf = true;
    }
    auto search(string word) {
        Trie* node = this->searchPrefix(word);
        return node != nullptr && node->isLeaf;
    }

    auto startsWith(string prefix) {
        return this->searchPrefix(prefix) != nullptr;
    }
};
```
例题：
```cpp
//oj 04089 电话号码
class Trie {
private:
    vector<Trie *> children;
    bool isLeaf;
public:
    Trie() : children(10), isLeaf(false) {}
    auto insert_and_check(string word) {//按题意修改
        Trie *node = this;
        for (auto c : word) {
            c -= '0';
            if (node->isLeaf) return false;
            if (node->children[c] == nullptr) node->children[c] = new Trie();
            node = node->children[c];
        }
        if (node->isLeaf) return false;
        for (int i = 0; i < 10; i++)
            if (node->children[i] != nullptr) return false;
        node->isLeaf = true;
        return true;
    }
};
int main() {
    int t;
    cin >> t;
    while (t--) {
        int n;
        cin >> n;
        bool check = false;
        vector<string> phone_num(n);
        for (int i = 0; i < n; i++)
            cin >> phone_num[i];
        sort(phone_num.begin(), phone_num.end());
        Trie *trie = new Trie();
        for (auto s : phone_num)
            if (!trie->insert_and_check(s)) {
                cout << "NO\n";
                goto next;
            }
        cout << "YES\n";
    next:;
    }
    return 0;
}
```

### 3.Manacher Algorithm
用于寻找回文串。
```cpp
void manacher(string t) {
	int r = 0, c = 0;
	for(int i = 0; i < n; i++) {
		if(i < r) p[i] = min(p[(c << 1) - i], r - i);
		else p[i] = 1;
		while(t[i - p[i]] == t[i + p[i]]) p[i]++;
		if(i + p[i] > r) {
			r = i + p[i];
			c = i;
		}
	}
}
```
例题：
```cpp
//Luogu P4555 [国家集训队] 最长双回文串
const int N = 1e5 + 5;
int n, ans, p[N << 1], l[N << 1], r[N << 1];
string s, t;
void manacher(string t) {
	int n = t.size();
	int R = 0, c = 0;
	for(int i = 0;i < n;i++) {
		if(i < R) p[i] = min(p[(c << 1) - i], R - i);
		else p[i] = 1;
		l[i - p[i] + 1] = max(l[i - p[i] + 1], p[i] - 1);
		r[i + p[i] - 1] = max(r[i + p[i] - 1], p[i] - 1);
		while(t[i - p[i]] == t[i + p[i]]) {
			p[i]++;
			l[i - p[i] + 1] = max(l[i - p[i] + 1], p[i] - 1);
			r[i + p[i] - 1] = max(r[i + p[i] - 1], p[i] - 1);
		}
		if(i + p[i] > R) {
			R = i + p[i];
			c = i;
		}
	}
}
int main() {
	cin >> s;
	n = s.size();
	t = "$#"; 
	for(int i = 0;i < n;i++) {
		t.push_back(s[i]);
		t.push_back('#');
	}
	t.push_back('&');//在string中插入'#', 最外面包两个'&'避免奇偶性讨论， 如：&#a#b#b#a#&
	manacher(t);
	n = t.size();
	for(int i = 3;i < n - 2;i++)
		if(t[i] == '#')
			ans = max(ans, l[i] + r[i]);
	cout << ans;
	return 0;
}
```

### 4.Sliding Window
例题：
```cpp
//sy10065. 独特蘑菇
int n, k; cin >> n >> k;
vector<int> c(n);
for (auto &i : c) cin >> i;
map<int, int> hash;
int l = 0; long long cnt = 0;
for (int r = 0; r < n; r++) {
    hash[c[r]]++;
    while (hash.size() > k) {
        hash[c[l]]--;
        if (!hash[c[l]]) hash.erase(c[l]);
        l++;
    }
    cnt += r - l + 1;
}
cout << cnt << '\n';
```
滑动窗口最大值问题：
deque 维护单调性：
用一个 单调递减队列（里面放的是下标，不是值）：

当右端加入新元素时，把所有比它小的元素都弹出（因为它们永远不会再成为最大值）；
当左端元素滑出窗口时，如果它正好是队首，就把它弹出。
这样，队首元素始终是当前窗口的最大值。

例题：
```cpp
//给你一个数组 nums 和一个整数 k，请输出每个长度为 k 的子数组的最大值。
vector<int> maxSlidingWindow(const vector<int> &nums, int k) {
    vector<int> ans;
    deque<int> dq;
    for (int r = 0; r < nums.size(); r++) {
        while (!dq.empty() && nums[dq.back()] <= nums[r]) dq.pop_back();
        dq.push_back(r);
        if (dq.front() <= r - k) dq.pop_front();
        if (r >= k - 1) ans.push_back(nums[dq.front()]);
    }
    return ans;
}
```

---
### IX.Other trivial things
#### 1.求解或判断质数
如果是判断某个数字或者很少的数字是否为质数，可用步长为6来判断（因为质数除了2，3都满足6k-1或6k+1）；
如果是判断较多数字是否为质数，或者获取大区间内的质数，使用欧拉筛
```cpp
//欧拉筛
const int MAXN = 100005;
int primes[MAXN], cnt = 0;
bool is_prime[MAXN]; // 初始化为 true
void euler_sieve(int n) {
    memset(is_prime, true, sizeof(is_prime));
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i <= n; ++i) {
        if (is_prime[i]) primes[cnt++] = i;
        for (int j = 0; j < cnt && i * primes[j] <= n; ++j) {
            is_prime[i * primes[j]] = false;
            if (i % primes[j] == 0) break; // 关键：保证每个数只被最小质因子筛去
        }
    }
}

```
---
#### 2.分解质因数
```cpp
vector<int> get_prime_factors(int n) {
    vector<int> factors;
    for (int i = 2; i * i <= n; ++i)
        while (n % i == 0) factors.push_back(i), n /= i;
    if (n > 1) factors.push_back(n);
    return factors;
}
```

---
### BASIC GRAMMAR
#### 2.二分查找
二分:
binary_search(bg, ed, val): 返回是否存在;
lower_bound(bg, ed, val): 第一个 >= val 的迭代器; upper_bound(bg, ed, val): 第一个 > val 的迭代器

例题：
```cpp
//最小值最大化问题 oj-aggressive cows-02456
int maximizeMinimum(vector<int> &list, int c) {
    int l = 1, r = list.back() - list[0], mid, ans;
    while (l < r) {
        mid = l + (r - l) / 2;
        int curr = list[0], cnt = 1;
        for (int i = 1; i < n; i++)
            if (list[i] - curr >= mid) {
                curr = list[i], cnt++;
                if (cnt >= c) break;
            }
        if (cnt >= c) ans = mid, l = mid + 1;
        else r = mid;
    }
    return ans;
}
```
```cpp
//最大值最小化问题 lc 1760 袋子里最少数目的球
int minimizeMaxium(vector<int> &nums, int maxOperations) {
    int l = 1, r = *max_element(nums.begin(), nums.end()), mid;
    while (l < r) {
        mid = l + (r - l) / 2;
        int cnt = 0;
        for (const auto &n : nums) {
            cnt += (n - 1) / mid;
            if (cnt > maxOperations) break;
        }
        if (cnt > maxOperations) l = mid + 1;
        else r = mid;
    }
    return l;
}
```
---
#### 3.排列组合
`next_permutation(v.begin(), v.end())`: 原地修改为下一个字典序排列。
```cpp
vector<int> v = {1, 2, 3};
do { /* process v */ } while (next_permutation(v.begin(), v.end()));
```
---
#### 4.矩阵乘法 
```cpp
typedef vector<vector<int>> matrix;
matrix multiply(const matrix& A, const matrix& B) {
    int n = A.size(), m = B[0].size(), k = A[0].size(); // A的列数 == B的行数
    matrix C(n, vector<int>(m, 0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            for (int p = 0; p < k; ++p)
                C[i][j] += A[i][p] * B[p][j];
    return C;
}
```