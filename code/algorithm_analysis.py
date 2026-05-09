import matplotlib.pyplot as plt
import timeit
import math
from functools import partial
import random

# 定义模拟不同复杂度的函数
def Constant(N):
    _ = 1

def Logarithmic(N):
    # 模拟 O(log N) 的操作次数
    _ = [i for i in range(int(math.log2(N)) if N > 0 else 0)]
 
def Linear(N):
    _ = [i for i in range(N)]

def N_Log_N(N):
    # 模拟 O(N log N) 的操作次数
    _ = [i for i in range(int(N * math.log2(N)) if N > 0 else 0)]
 
def Quadratic(N):
    _ = [i for i in range(int(math.pow(N, 2)))]
 
def Cubic(N):
    _ = [i for i in range(int(math.pow(N, 3)))]

def Exponential(N):
    # 指数增长极快，N稍大即会导致内存溢出或超时
    _ = [i for i in range(int(math.pow(2, min(N, 20))))] 

def plotTC(fn, nMin, nMax, nInc, nTests):
    """
    运行计时器并绘制时间复杂度曲线
    """
    x, y = [], []
    for i in range(nMin, nMax, nInc):
        testNTimer = timeit.Timer(partial(fn, i))
        t = testNTimer.timeit(number=nTests)
        x.append(i)
        # 对运行时间取对数，以便在同一张图中观察跨度巨大的函数
        y.append(math.log2(t) if t > 0 else 0)

    plt.plot(x, y, label=fn.__name__)
    plt.legend(loc='best')
 
def main():
    print('Analyzing Algorithms...')
    upbound = 100
    step = 1
    # 排除 Exponential 是因为其在 N=100 时会产生天文数字级的计算量
    functions = [Constant, Logarithmic, Linear, N_Log_N, Quadratic, Cubic]
    
    for func in functions:
        plotTC(func, 10, upbound, step, 10)
 
    plt.title("Growth Rates of Common Functions (Log Scale Time)")
    plt.xlabel("Input Size (n)")
    plt.ylabel("Time Complexity (log2 of time)")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
