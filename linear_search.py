# -*- coding: utf-8

'''采用进退法来确定搜索区间,采用0.618搜索最优的alpha,然后采用最速梯度下降法找到最优解
====================================================
矩阵的条件数越接近于1的话,那么最速下降法的收敛速率就越快||
====================================================
'''

from math import sqrt
import time
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt

def objfunc(var_x):
    '''定义好一个指定的函数: X'AX + b'X + 4
    该函数是定义在n维空间上,映射到一维空间上

    Arguments:
        x {float型列向量} -- 函数的自变量

    Returns:
        float -- 函数的返回值
    '''
    var_x = np.array(var_x)  # 将x转为array,方便进行矩阵运算
    # 该矩阵不是标准二次型,但是正定的,也就是说函数一定是凸函数,这确保了搜索出的近似局部最优点一定是全局最优点
    matrix_a = np.array([[2., -2.], [-2., 4.]])
    cmatrix_b = np.array([[-4.], [0.]])
    # 先求转置,再求点积
    return var_x.T.dot(matrix_a).dot(var_x) + cmatrix_b.T.dot(var_x) + 4


def grad(var_x):
    '''

    计算函数的梯度函数

    Arguments:
        var_x {float列向量} -- 函数自变量

    Returns:
        列向量 -- 当前点的梯度向量
    '''
    var_x = np.array(var_x)
    matrix_a = np.array([[2., -2.], [-2., 4.]])
    cmatrix_b = np.array([-4., 0.])
    return 2. * matrix_a.dot(var_x) + cmatrix_b


def search_interval(func, start, step_h, gamma, itr=3000):
    '''[精确线性搜索法, 关于alpha的函数,一维函数的最优化问题,目的是寻找出近似的高峰搜索区间 -- 这是alpha的可选区间]

    [采用进退法来确定搜索区间,这是一个不需要导数的搜索方法,过程中是已知函数的下降方向,一般取梯度的反方向]

    Arguments:
        func {object} -- [确定步长的一元函数,看成是关于 alpha 的函数]
        start {float} -- [初始值]
        h {float} -- [确定搜索区间的进退法的步长]
        gamma {[float} -- [进退法中的一个常数,用来增大步长进行搜索,大于1]
        itr {int} -- [迭代上限]
        '''
    point_a0 = start
    value0 = func(point_a0)
    point_a1 = point_a0 + step_h
    value1 = func(point_a1)
    sgn = 1  # 符号变量,如果一开始 v1 >= v0 ,那么就取 sgn = -1,反向搜索
    # 一开始通过比较第一组值来判断到底是正向搜索还是反向搜索, 确保是下降方向
    # 如果后面发现又有一个反向的上升,那么就退出循环
    if value0 <= value1:
        point_a0, point_a1 = point_a1, point_a0
        # 交换值进行反向搜索,总之就是确保底数的增加是按函数的下降方向,便于进行值的更新
        value0, value1 = value1, value0
        sgn = -1
    point_a2 = point_a1
    k = 1   # 表示迭代次数
    while k < itr:
        point_a2 += sgn * (gamma**k) * step_h
        # 如果一开始 v1 < v0 ,那么就是正向增大的搜索, gamma值较小,
        # 使得越到和后面得增量越小, 这对于初始点的选取要求很高,
        # 因为如果初始点离最优点的距离越远,那么迭代次数会非常庞大
        value2 = func(point_a2)
        if value2 > value1:
            # 此时已经满足一个高-低-高的高峰情况,在函数连续可微的情况下是肯定存在局部最优点的
            # 注意此时方向决定了 a0 与 a2 的大小关系
            return (point_a0, point_a2) if sgn == 1 else (point_a2, point_a0)
        point_a0 = point_a1
        point_a1 = point_a2
        value1 = value2   # 点与函数值移动
        k += 1
    print('Cannot find a valid interal')

# ====================================================================
# 该算法会根据搜索区间的长度进行步长的动态计算,所以不管一开始的初始点有多大,
# 经过几次的搜寻总能立即进入近似的收敛的情况,虽然可能迭代的次数较多,但是每一
# 次只需要更新一个端点,而且步长较大,足以弥补迭代次数多的缺点
# ====================================================================


def direct_search(
        func, a, b, e, al=None, ar=None, lvalue=None, rvalue=None):
    '''[采用 0.618 法在搜索区间寻找局部最优点]

    [该方法是一个不依赖于导数的精确线性搜索方法]

    Arguments:
        func {object} -- [目标函数]
        a {float} -- [搜索区间的左端点]
        b {float} -- [搜索区间的右端点]
        e {float} -- [允许误差范围,用于控制函数的退出条件]

    Keyword Arguments:
        al {float} -- [0.618 取点时的左端点] (default: {None})
        ar {float} -- [0.618 取点时的右端点] (default: {None})
        lvalue {float} -- [对应于al的函数值] (default: {None})
        rvalue {float} -- [对应于ar的函数值] (default: {None})
        '''

    interval = (sqrt(5) - 1) / 2  # 控制间距的常量,方便每次只更新一个值
    if al is None:
        al = a + (1 - interval) * (b - a)  # 要求每一次的区间长度的缩减都是一致的
    if ar is None:
        ar = a + interval * (b - a)
    if lvalue is None:
        lvalue = func(al)
    if rvalue is None:
        rvalue = func(ar)
    if abs(b - a) < e:  # 退出 0.618 循环, 返回的是精确线性搜索找到的对应 Xk处的系数alpha
        res = al if lvalue < rvalue else ar
        return res
    if lvalue < rvalue:
        b = ar
        ar = al
        rvalue = lvalue
        return direct_search(func, a, b, e, ar=ar, rvalue=rvalue)
    a = al
    al = ar
    lvalue = rvalue
    return direct_search(func, a, b, e, al=al, lvalue=lvalue)

# 非精确一位搜索方法
# 最速下降法
# 每一次迭代都要寻找一次最优的alpha,然后不断迭代当前点的负梯度方向


def steepest_grad(func, objg, start, e, X, Y, itertimes):
    '''[最速梯度下降法]

    [对每一个点处都进行进退法得到搜索区间, 线性搜索找到近似最优的alpha系数, 由于选定的是负梯度方向,所以叫精确线性搜索]

    Arguments:
        func {object} -- [目标函数]
        grad {object} -- [梯度函数]
        start {list} -- [自变量的初始点]
        e {float} -- [允许的梯度的误差范围]
        tmp {float} -- [用于记录alpha的值]
        note_ {int} -- [用于记录迭代次数]

    Returns:
        [(tmp, start)] -- [返回的是最终的alpha值, 局部最优点的坐标]
        '''

    begin = time.clock()
    X.append(start[0])
    Y.append(start[1])

    if sum(objg(start)**2) < e**2:     # 梯度的范数小于给定的误差范围就退出最速梯度下降
        end = time.clock()
        interp = end - begin
        itertimes += 1
        print("{0:^10d}{1:^20.5f}{2:^20.5f}{3:^20.5f}".format(
            itertimes, 0., norm(objg(start), 2), interp))
        return start     # 返回的是(系数点, 自变量Xk)

    cur_grad = -objg(start)    # 负梯度方向
    # 初始点为0, 步长为0.1, gamma = 2.
    left, right = search_interval(
        (lambda a: func(start + a * cur_grad)), 1000., 0.2, 2)
    alpha = direct_search(
        (lambda alpha: func(start + alpha * cur_grad)), left, right, 1e-4)

    # string = '{0:^10.5f}[{1:^10.5f},{2:^10.5f}]{3:^10.10f}'.format(
    #     note_a, start[0], start[1], res)

    end = time.clock()
    interp = end - begin
    itertimes += 1
    print("{0:^10d}{1:^20.5f}{2:^20.5f}{3:^20.5f}".format(
        itertimes, alpha, norm(objg(start), 2), interp))
    # sys.stdout.write(interp)
    return (
        steepest_grad(func, objg, start + alpha * cur_grad, e, X, Y,
                      itertimes))


if __name__ == "__main__":
    X = []
    Y = []
    ITER = 0
    print("这是最速梯度算法得基本指标:")
    print('=' * 78)
    print("{0:^10s}{1:^20s}{2:^20s}{3:^20s}".format(
        "itertimes", "alpha", "norm_g", "time"))
    print('=' * 78)

    SOLUTION = steepest_grad(objfunc, grad, [0., 3.], 3e-5, X, Y, ITER)
    print("函数得最值点为: [ %-15.5f , %-15.5f ]" % (SOLUTION[0], SOLUTION[1]))
    print("函数的近似最小值为: %.5f" % objfunc(SOLUTION))
    E_GRAD = grad(SOLUTION)
    print("此时函数的梯度为: [ %020.5f , %020.5f ]" % (E_GRAD[0], E_GRAD[1]))
    print("此时的梯度范数为: %010.5f" % np.linalg.norm(E_GRAD))

    plt.plot(X, Y)

    X = np.arange(-5, 5, 0.01)
    Y = np.arange(-5, 5, 0.01)
    [X, Y] = np.meshgrid(X, Y)
    Z = 2 * (X**2 - 2 * X * Y + 2 * Y**2) - 4 * X + 4
    plt.contourf(X, Y, Z)
    plt.show()
