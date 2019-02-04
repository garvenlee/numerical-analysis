# -*- coding:utf-8 -*-

'''
   @Author     : garvenlee
   @DateTime   : 2018-12-30 11:08:47
   @Version    : python 3.6.5

'''

import time
from numpy.linalg import norm
from numpy import (array, Inf, delete, random, zeros,
                   eye, ones, sqrt, hstack, vstack, argmin)
import matplotlib.pyplot as plt
from functools import partial
from scipy.linalg import inv, qr

eps = 1e-5
m = 5
n = 10
A = random.randn(m, n)
A = A.T.dot(A) + 0.01 * eye(n, n)
c = random.randn(n, 1)

_all = [
    'steepest_decent', 'conjugate_gradient',
    'powell',
    'linear_newton_CG', 'bfgs', 'lbfgs',
    'trust_region(cauchy_point,dogleg)',
    'LM', 'gauss_newton',
    'active_set', 'null_space', 'lagrange',
    'feasible_direction', 'rosen',
    'multiplier_method', 'newton_lagrange', 'SQP'
]
color = ['r', 'b', 'k', 'y', 'g']


def objfunc(x):
    x = array(x)
    return float(0.5 * x.T.dot(A.dot(x)) + c.T.dot(x))


def grad(x):
    return A.dot(x) + c


def hesse(x):
    return A


def column_name(flag):
    '''用于打印迭代列名

    Args:
        flag: value to distinguish linear search and trust region
            <0:linear search    1:trust region>
    '''
    if flag == 0:
        print("{0:<6s}{1:^10s}{2:^10s}{3:^15s}{4:^15s}".format(
            "iter", "alpha", "norm_dk", "fvalue", "time"))
    else:
        print("{0:<6s}{1:^10s}{2:^15s}{3:^15s}".format(
            "iter", "norm_dk", "fvalue", "time"))


def wolfe(f, grad, xk, dk, c1=0.25, c2=0.75, maxiter=20):
    '''使用 wolfe 准则进行非精确线搜索

    Args:
        f: object function
        grad: the gradient of the object function
        xk: current iter point 'xk'
        dk: current search direction 'dk'
        c1: args for wolfe condition (default: {0.25})
        c2: args for wolfe condition (default: {0.75})
        maxiter: the max iter times (default: {20})
    '''
    alpha = 1
    a = 0
    b = Inf
    k = 0
    fk = f(xk)
    gdk = grad(xk).T.dot(dk)
    while k < maxiter:
        fk_1 = f(xk + alpha * dk)
        if fk_1 >= fk + c1 * alpha * gdk:
            b = alpha
            alpha = (alpha + a) / 2
            continue
        if fk_1.T.dot(dk) < c2 * gdk:
            a = alpha
            alpha = min([2 * alpha, (b + alpha) / 2])
            continue
        break


def armijio(f, gk, xk, dk, beta=0.9, sigma=1e-4, maxiter=20):
    '''非精确线搜索中的Armijio准则

    Args:
        f: object function
        gk: gradient of object function at 'xk'
        xk: current point 'xk'
        dk: current search direction 'dk'
        beta: args for alpha (default: {0.9})
        sigma: args for armijio (default: {1e-4})
        maxiter: the max iter times (default: {20})


    Returns:
        [description] the step 'alpha'
        [type] float
    '''
    k = 0
    fk = f(xk)
    while k <= maxiter:
        alpha = beta**k
        if f(xk + alpha * dk) <= fk + sigma * alpha * float(gk.T.dot(dk)):
            break
        k += 1
    if alpha < 1e-6:
        alpha = 1.
    return alpha


def search_interval(func, start, step_h, gamma, itr=25):
    '''[精确线性搜索法, 关于alpha的函数,一维函数的最优化问题,目的是寻找出近似的高峰搜索区间 -- 这是alpha的可选区间]

    [采用进退法来确定搜索区间,这是一个不需要导数的搜索方法,过程中是已知函数的下降方向,一般取梯度的反方向]

    Arguments:
        func {object} -- [确定步长的一元函数,看成是关于 alpha 的函数]
        start {float} -- [初始值]
        step_h {float} -- [确定搜索区间的进退法的步长]
        gamma {[float} -- [进退法中的一个常数,用来增大步长进行搜索,大于1]
        itr {int} -- [迭代上限]
        '''
    point_a0 = start
    value0 = func(point_a0)
    point_a1 = point_a0 + step_h
    value1 = func(point_a1)
    sgn = 1  # 符号变量,如果一开始 v1 >= v0 ,那么就取 sgn = -1,反向搜索

    if value0 <= value1:
        point_a0, point_a1 = point_a1, point_a0
        value0, value1 = value1, value0
        sgn = -1
    point_a2 = point_a1
    k = 1
    while k < itr:
        point_a2 += sgn * (gamma**k) * step_h
        value2 = func(point_a2)
        if value2 > value1:
            break
        point_a0 = point_a1
        point_a1 = point_a2
        value1 = value2   # 点与函数值移动
        k += 1
    # print('Cannot find a valid interal')
    return (point_a0, point_a2) if sgn == 1 else (point_a2, point_a0)


def direct_search(
        func, a, b, e, al=None, ar=None, lvalue=None, rvalue=None, maxiter=25):
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
    k = 0
    interval = (sqrt(5) - 1) / 2
    while k < maxiter:
        if al is None:
            al = a + (1 - interval) * (b - a)  # 要求每一次的区间长度的缩减都是一致的
        if ar is None:
            ar = a + interval * (b - a)
        if lvalue is None:
            lvalue = func(al)
        if rvalue is None:
            rvalue = func(ar)
        if abs(b - a) < e:
            res = al if lvalue < rvalue else ar
            return res
        if lvalue < rvalue:
            b = ar
            ar = al
            rvalue = lvalue
            al = None
            lvalue = None
            continue
        a = al
        al = ar
        lvalue = rvalue
        ar = None
        lvalue = None
        continue
    pnt = ar if al is None else al
    return pnt if pnt < b else (a + b) / 2


def q_interpolation(f, a, b, maxiter=25, delta=1e-4, eps=1e-6):
    '''使用抛物线法二次插值

    Args:
        f: 单值函数
        a: 搜索区间左端点
        b: 搜索区间右端点
        maxiter: 最大迭代次数 (default: {25})
        delta: 最小搜索步长 (default: {1e-4})
        eps: 误差终止数 (default: {1e-6})

    Returns:
        [description]返回最优搜索点
        [type]float
    '''
    k = 0
    cond = 0
    err = 1
    h = 1
    s0 = a
    ds = 1e-5
    if abs(s0) > 1e-4:
        h = abs(s0) * 1e-4
    while k < maxiter and err > eps and not(cond == 5):
        # 判断当前迭代点 s0 处的斜率走向，确定三点的布局
        df = (f(s0 + ds) - f(s0 - ds)) / (2 * ds)
        if df > 0:
            h = -abs(h)
        bars = s0
        s1 = s0 + h
        s2 = s0 + 2 * h
        f0 = f(s0)
        f1 = f(s1)
        f2 = f(s2)
        barf = f0
        cond = 0
        j = 0
        # 找到单峰区间
        while j < maxiter and abs(h) > delta and cond == 0:
            if f0 <= f1:
                s2 = s1
                f2 = f1
                h = 0.5 * h
                s1 = s0 + h
                f1 = f(s1)
            else:
                if f2 < f1:
                    s1 = s2
                    f1 = f2
                    h = 2 * h
                    s2 = s0 + 2 * h
                    f2 = f(s2)
                else:
                    cond = -1
            j += 1
            if abs(h) > 1e6 or abs(s0) > 1e6:
                cond = 5
        if cond == 5:
            bars = s1
            barf = f(s1)
        else:
            # 插值更新迭代
            d = 2 * (2 * f1 - f0 - f2)
            if d < 0:
                barh = h * (4 * f1 - 3 * f0 - f2) / d
            else:
                barh = h / 3
                cond = 4
            bars = s0 + barh
            barf = f(bars)
            h = abs(h)
            h0 = abs(barh)
            h1 = abs(barh - h)
            h2 = abs(barh - 2 * h)
            if h0 < h:
                h = h0
            if h1 < h:
                h = h1
            if h2 < h:
                h = h2
            if h == 0:
                h = barh
            if h < delta:
                cond = 1
            if abs(h) > 1e6 or abs(bars) > 1e6:
                cond = 5
            err = abs(f1 - barf)
            s0 = bars
        if cond == 2 and h < delta:
            cond = 3
        k += 1
    return s0 if s0 < b else (a + b) / 2


def steepest_decent(f, grad, xk, c1=None, c2=None, beta=None,
                    sigma=None, maxiter=3000, maxiter_linear=None,
                    exact=False, exact_flag=0, disp=True,
                    plot_flag=False, momentum_flag=False,
                    momentum=None):
    '''使用最速梯度下降法求解目标函数的最优解

    Args:
        f: object function
        grad: gradient of object function
        xk: current point 'xk'
        c1: given args for wolfe condition (default: {None})
        c2: given args for wolfe condition (default: {None})
        beta: given args for armijio condition (default: {None})
        sigma: given args for armijio condition (default: {None})
        maxiter: the max iter times (default: {2000})
        maxiter_linear: the max iter times for linear search (default: {None})
        exact: bool value to decide to use exact linear search
            (default:{False})
        exact_flag: decide to use interpolation or 0.618
            (default:{0}) <0:q_interpolation, 1:direct_search>
        disp: bool value to decide to print iter info
            (default: {False})    <x:xk[0]  y:f(xk)>
        plot_flag: bool value to decide to plot all points searched
            or not (default: {False})    <x:xk[0]  y:f(xk)>
        momentum_flag: bool value to decide to use momentum to accelerate
            (default: {False})
        momentum: args for momentum accelerating (default: {None})
            between 0 and 1

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    cmp = None
    value = None
    k = 0
    change_value = None
    if disp:
        column_name(0)
        begin = time.clock()

    if plot_flag:
        points = []
        point = []
        m = 5
        for i in range(m + 1):
            points.append(point.copy())
    while k < maxiter:
        gk = grad(xk)
        dk = -gk
        norm_dk = norm(dk)
        fk = f(xk)

        if norm_dk < eps:
            break

        if not (cmp is None):
            if abs(cmp - norm_dk) < 0.1 * eps or abs(value - fk) < 0.1 * eps:
                break
        cmp = norm_dk
        value = fk

        if plot_flag:
            for i in range(m):
                points[i].append(xk[i])
            points[-1].append(fk)

        if not exact:
            if not (c1 is None) and not (c2 is None):
                if maxiter_linear is None:
                    alpha = wolfe(f, gk, xk, dk, c1, c2)
                else:
                    alpha = wolfe(f, gk, xk, dk, c1, c2,
                                  itremax=maxiter_linear)
            elif not (beta is None) and not (sigma is None):
                if maxiter_linear is None:
                    alpha = armijio(f, gk, xk, dk, beta, sigma)
                else:
                    alpha = armijio(f, gk, xk, dk, beta,
                                    sigma, maxiter=maxiter_linear)
            else:
                if maxiter_linear is None:
                    alpha = armijio(f, gk, xk, dk)
                else:
                    alpha = armijio(f, gk, xk, dk, maxiter=maxiter_linear)
        else:
            if exact_flag:
                left, right = search_interval(
                    (lambda alpha: f(xk + alpha * dk)), 0.1, 0.01, 2)
                alpha = direct_search(
                    (lambda alpha: f(xk + alpha * dk)), left, right, 1e-5)
            else:
                alpha = q_interpolation(
                    (lambda alpha: f(xk + alpha * dk)), 0, 1, 1e-5)

        if momentum_flag:
            if change_value is None:
                xk = xk + alpha * dk
                change_value = alpha * dk
            else:
                xk = xk + alpha * dk + momentum * change_value
                change_value = alpha * dk + momentum * change_value
        else:
            xk = xk + alpha * dk
        k += 1

        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:15.4f}{4:15.6f}".format(
                k, alpha, norm_dk, fk, end - begin))

    if plot_flag:
        fig = plt.figure()
        ax = fig.gca()
        for i in range(m):
            ax.plot(points[i], points[-1], color[i])
        plt.show()

    print("共迭代{0}次,最优解为{1:.4f}".format(k, fk))
    return xk


def conjugate_gradient(f, grad, hesse, xk, maxiter=3000,
                       maxiter_linear=None, exact=False,
                       plot_flag=False, disp=False):
    '''使用CG算法求解最优化问题 0.5*x'Ax + b'x + c

    Args:
        f: object function
        grad: the gradient of object function
        xk: current iter point 'xk'
        maxiter: the max iter times (default: {3 * n})
        maxiter_linear: the max iter times for armijio search
            (default: {None})
        exact: bool value to decide to search exactly (default: {False})
        plot_flag: bool value to decide to plot or not  (default: {False})
        disp: bool value to decide to print iter info or not
            `(default: {False})

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    cmp = None
    value = None
    if exact:
        G = hesse(xk)
    if disp:
        column_name(0)
        begin = time.clock()
    k = 0
    rk = grad(xk)
    norm_rk = norm(rk)
    dk = -rk
    if plot_flag:
        point = []
        points = []
        for i in range(m + 1):
            points.append(point.copy())
    while k < maxiter:
        fk = f(xk)
        if norm_rk < eps:
            break

        if not (cmp is None):
            if abs(cmp - norm_rk) < 0.1 * eps or abs(value - fk) < 0.1 * eps:
                break
        cmp = norm_rk
        value = fk

        if plot_flag:
            for i in range(m):
                points[i].append(xk[i])
            points[-1].append(fk)
        if not exact:
            if maxiter_linear is None:
                alpha = armijio(f, rk, xk, dk)
            else:
                alpha = armijio(f, rk, xk, dk, maxiter=maxiter_linear)
        else:
            alpha = float(norm_rk**2 / (dk.T.dot(G.dot(dk))))
        xk = xk + alpha * dk
        rk_1 = grad(xk)
        norm_rk_1 = norm(rk_1)
        beta = (norm_rk_1 / norm_rk)**2
        dk = -rk_1 + beta * dk
        rk = rk_1
        norm_rk = norm_rk_1
        k += 1

        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^15.4f}{4:^15.6f}".format(
                k, alpha, norm_rk, fk, end - begin))

    if plot_flag:
        fig = plt.figure()
        ax = fig.gca()
        for i in range(m):
            ax.plot(points[i], points[-1], color[i])
        plt.show()

    print("共迭代{0}次,最优解为{1:.4f}".format(k, fk))
    return xk


def powell(f, xk, maxiter=25, maxiter_linear=None, disp=False, err=1e-5):
    '''powell 算法不需要梯度求解凸最优化问题

    Args:
        f: object function
        xk: current iter point 'xk'
        maxiter: the max iter times (default: {5})
        maxiter_linear: the max iter times for linear search (default: {None})

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    k = 0
    n = xk.shape[0]

    if maxiter_linear is None:
        maxiter_linear = 25

    if disp:
        begin = time.clock()
        print("{0:<6s}{1:^15s}{2:^15s}".format('iter', 'f_value', "time"))

    while k < maxiter:
        Ik = eye(n)
        former = xk
        fk = f(xk)
        xpos = []
        xpos.append(xk)
        for i in range(n):
            vk = Ik[i].reshape((n, -1))
            left, right = search_interval(
                (lambda alpha: f(xpos[i] + alpha * vk)), 0, 0.01, 1.1,
                itr=maxiter_linear)
            alpha = direct_search(
                (lambda alpha: f(xpos[i] + alpha * vk)), left, right,
                1e-5, maxiter=maxiter_linear)
            # alpha = q_interpolation(
            #     (lambda alpha: f(xpos[i] + alpha * vk)), left, right)

            xpos.append(xpos[i] + alpha * vk)

        Ik = delete(Ik, 0, 0)
        vk = xpos[-1] - xpos[0]
        Ik = vstack(((Ik, vk.reshape((-1, n)))))

        left, right = search_interval(
            (lambda alpha: f(xpos[-1] + alpha * vk)), 0, 0.01, 1.1,
            itr=maxiter_linear)
        alpha = direct_search(
            (lambda alpha: f(xpos[-1] + alpha * vk)), left, right,
            1e-5, maxiter=maxiter_linear)
        # alpha = q_interpolation(
        #     (lambda alpha: f(xpos[-1] + alpha * vk)), left, right)
        xk = xpos[-1] + alpha * vk

        if disp:
            end = time.clock()
            print('{0:<6d}{1:^15.4f}{2:^15.5f}'.format(k, fk, end - begin))

        if abs(f(xk) - fk) < err or abs(norm(xk - former)) < err:
            break

        k += 1

    if disp:
        print("共迭代{0}次,最优解为{1:.4f}".format(k, fk))
    return xk


def CG_modification(Bk, gk, maxiter=5, exact=True, f=None,
                    grad=None, modify_flag=False, tao=0.5, eps=None):
    '''修正的CG算法,用于应对矩阵不正定的情况
    求解 Bk.x = -gk

    修正有两种方法: 1.返回最速下降方向
            2.修改系数矩阵,使成为正定矩阵

    Args:
        Bk: 系数矩阵
        gk: 当前迭代点的梯度方向
        maxiter: 最大迭代次数 (default: {5})
        modify_flag: 是否修正系数矩阵 (default: {False})
        tao: 修正系数矩阵的参数 (default: {0.5})
            推荐是0-1之间的数

    Returns:
        [description] 返回搜索方向
        [type] 列向量
    '''
    n = gk.shape[0]
    zk = zeros((n, 1))
    rk = gk
    dk = -gk
    norm_gk = norm(gk)
    norm_rk = norm_gk

    if eps is None:
        epsk = min(0.5, sqrt(norm_gk)) * norm_gk
    else:
        epsk = eps
    k = 0
    while k < maxiter:
        if norm_rk < epsk:
            return zk
        Bd = Bk.dot(dk)
        dBd = float(dk.T.dot(Bd))
        if dBd <= 0:
            if not modify_flag:
                if k == 0:
                    return -gk
                else:
                    return zk
            else:
                Bk = Bk + tao * eye(n, n)
                continue

        if not exact:
            alpha = armijio(f, rk, zk, dk)

        else:
            alpha = norm_rk**2 / dBd

        zk = zk + alpha * dk

        if not exact:
            rk = grad(zk)
        else:
            rk = rk + alpha * Bd

        norm_rk_1 = norm(rk)

        beta = (norm_rk_1 / norm_rk)**2
        dk = -rk + beta * dk
        norm_rk = norm_rk_1

        k += 1
    return zk


def linear_newton_CG(f, grad, hesse, xk, plot_flag=False,
                     disp=False, exact=False, modify_flag=False,
                     maxiter=3000, cg_maxiter=None, maxiter_linear=None):
    '''线搜索 newton-cg 方法求解最小化问题

    Args:
        f: object function
        grad: the gradient of the object function
        hesse: function to get the hessian matrix
        xk: vurrent iter point 'xk'
        plot_flag: bool value to decide to plot or not (default: {False})
        disp: bool value to decide to print iter info (default: {False})
        exact: bool value to decide to use exact linear search
             (default: {False})
        modify_flag: bool value to decide to modify matrix (default: {False})
        maxiter: the max iter times (default: {3 * n})
        cg_maxiter: the max iter times for modify_cg code (default: {None})

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    k = 0
    cmp = None
    value = None
    if disp:
        column_name(0)
        begin = time.clock()

    while k < maxiter:
        gk = grad(xk)
        hessian = hesse(xk)
        fk = f(xk)
        norm_gk = norm(gk)
        if norm_gk < eps:
            break
        if not (cmp is None):
            if abs(cmp - norm_gk) < 0.1 * eps or abs(value - fk) < 0.1 * eps:
                break
        cmp = norm_gk
        value = fk

        if not modify_flag:
            if cg_maxiter is None:
                pk = CG_modification(hessian, gk, f=f, grad=grad)
            else:
                pk = CG_modification(
                    hessian, gk, maxiter=cg_maxiter, f=f, grad=grad)
        else:
            if cg_maxiter is None:
                pk = CG_modification(
                    hessian, gk, modify_flag=True, f=f, grad=grad)
            else:
                pk = CG_modification(
                    hessian, gk, modify_flag=True,
                    maxiter=cg_maxiter, f=f, grad=grad)

        if not exact:
            if maxiter_linear is None:
                alpha = armijio(f, gk, xk, pk)
            else:
                alpha = armijio(f, gk, xk, pk, maxiter=maxiter_linear)
        else:
            left, right = search_interval(
                (lambda alpha: f(xk + alpha * pk)), 0.1, 0.01, 2)
            alpha = direct_search(
                (lambda alpha: f(xk + alpha * pk)), left, right, 1e-5)
        xk = xk + alpha * pk

        k += 1

        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^15.4f}{4:^15.6f}".format(
                k, alpha, norm_gk, fk, end - begin))

    print("共迭代{0}次,最优解为{1:.4f}".format(k, fk))
    return xk


def bfgs(f, grad, xk, plot_flag=False, disp=False,
         maxiter=3000, maxiter_linear=None):
    '''使用 BFGS 求解无约束最下化问题

    Args:
        f: object function
        grad: the gradient of object function
        xk: current iter point 'xk'
        plot_flag: bool value to decide to plot or not (default: {False})
        disp: bool value to decide to print iter info (default: {False})
        maxiter: the max iter times (default: {3 * n})
        maxiter_linear: the max iter times for linear search
                 (default: {None})

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    n = xk.shape[0]
    Hk = eye(n)
    Ik = eye(n)
    gk = grad(xk)

    cmp = None
    value = None
    k = 0
    if disp:
        column_name(0)
        begin = time.clock()
    while k < maxiter:
        norm_gk = norm(gk)
        fk = f(xk)

        if not (cmp is None):
            if abs(cmp - norm_gk) < 0.1 * eps or abs(value - fk) < 0.1 * eps:
                break
        cmp = norm_gk
        value = fk

        if norm_gk < eps:
            break
        dk = -1. * Hk.dot(gk)
        if maxiter_linear is None:
            alpha = armijio(f, gk, xk, dk)
        else:
            alpha = armijio(f, gk, xk, dk, maxiter=maxiter_linear)
        xk_1 = xk + alpha * dk
        gk_1 = grad(xk_1)

        sk = xk_1 - xk
        yk = gk_1 - gk
        ys = yk.T.dot(sk)
        if ys > 0:
            rhok = 1. / ys
            Vk = Ik - rhok * sk.dot(yk.T)
            Hk = Vk.dot(Hk).dot(Vk.T) + rhok * sk.dot(sk.T)
        gk = gk_1
        xk = xk_1
        k += 1
        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^15.4f}{4:^15.6f}".format(
                k, alpha, norm_gk, fk, end - begin))

    print("共迭代{0}次,最优解为{1:.4f}".format(k, fk))
    return xk


def lbfgs_two_loop(Hk, gk, s, y):
    '''使用lbfgs两步循环法求解当前的搜索方向

    Args:
        Hk: the initial hessian matrix
        gk: the gradient of object function
        s: list of vector s
        y: list of vector y
    '''
    q = gk
    length = len(s)
    rho_list = []
    alpha_list = []
    for i in range(length):
        rhok = 1. / ((y[length - i - 1].T).dot(s[length - i - 1]))
        rho_list.append(rhok)
        alpha = rhok * (s[length - i - 1].T).dot(q)
        alpha_list.append(alpha)
        q = q - alpha * y[length - i - 1]
    r = Hk.dot(q)
    for i in range(length):
        beta = rho_list[length - i - 1] * (y[i].T).dot(r)
        r = r + s[i] * (alpha_list[length - i - 1] - beta)
    return -r


def lbfgs(f, grad, xk, limited_num=None, disp=False,
          plot_flag=False, maxiter=3000, maxiter_linear=None):
    '''使用 lbfgs 内存优化算法求解无约束最小化问题

    Args:
        f: object function
        grad: the gradient of object function
        xk: current iter point 'xk'
        limited_num: note the formmmer vector (default: {int(n**(1 / 3))})
        disp: bool value to decide to print iter info (default: {False})
        plot_flag: bool value to decide to plot (default: {False})
        maxiter: the max iter times (default: {n})
        maxiter_linear: the max iter times for linear search (default: {None})

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    n = xk.shape[0]
    if limited_num is None:
        limited_num = int((n)**(1 / 3))
    cmp = None
    value = None
    Hk0 = eye(n)
    k = 0
    gk = grad(xk)
    s = []
    y = []
    if disp:
        column_name(0)
        begin = time.clock()

    while k < maxiter:
        norm_gk = norm(gk)
        fk = f(xk)

        if norm_gk < eps:
            break

        if not (cmp is None):
            if abs(cmp - norm_gk) < 0.1 * eps or abs(value - fk) < 0.1 * eps:
                break
        cmp = norm_gk
        value = fk
        if len(s):
            dk = lbfgs_two_loop(Hk0, gk, s, y)
        else:
            dk = -Hk0.dot(gk)
        alpha = armijio(f, gk, xk, dk, maxiter=100)

        xk_1 = xk + alpha * dk
        gk_1 = grad(xk_1)

        sk = xk_1 - xk
        yk = gk_1 - gk

        if k > limited_num:
            s.pop(0)
            y.pop(0)

        s.append(sk)
        y.append(yk)
        gk = gk_1
        xk = xk_1
        k += 1
        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^15.4f}{4:^15.6f}".format(
                k, alpha, norm_gk, fk, end - begin))

    print("共迭代{0}次,最优解为{1:.4f}".format(k, fk))
    return xk


def trust_region(f, grad, hesse, xk, delta=5, max_delta=10, eta=0.05,
                 mu1=0.25, mu2=0.75, method_flag=None, disp=False,
                 plot_flag=False, maxiter=3000):
    '''使用信頼域方法求解无约束最优化问题

    Args:
        f: object function
        grad: function to get the gradient of object function
        hesse: function to get the hessian matrix of object function
        xk: current iter point 'xk'
        delta: current trust region radius (default: {5})
        max_delta: the max trust region radius (default: {10})
        eta: args for choosing whether to change xk  (default: {0.05})
        mu1: args for changing trust region radius (default: {0.25})
        mu2: args for changing trust region radius (default: {0.75})
        method_flag: flag to select which trust region method
            (default: {None})   <0:dogleg   1:cauchy point>
        disp: bool value to decide to print iter info (default: {False})
        plot_flag: bool value to decide to plot (default: {False})
        maxiter: the max iter times (default: {20 * n})

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    cmp = None
    value = None
    if disp:
        column_name(1)
        begin = time.clock()
    k = 0
    while k < maxiter:
        gk = grad(xk)
        Bk = hesse(xk)
        fk = f(xk)
        norm_gk = norm(gk)
        if not(cmp is None):
            if abs(cmp - norm_gk) < eps or abs(value - fk) < eps:
                break
        cmp = norm_gk
        value = fk
        if norm_gk < eps:
            break
        if method_flag == 1:
            dk = cauchy_point(gk, Bk, delta)
        elif method_flag == 0 or method_flag is None:
            dk = dogleg(f, gk, Bk, delta, grad)
        else:
            return

        xk_1 = xk + dk
        fk_1 = f(xk_1)
        try:
            rhok = -(fk - fk_1) / float(gk.T.dot(dk) +
                                        0.5 * dk.T.dot(Bk.dot(dk)))
        except Exception:
            delta = min(2 * delta, max_delta)
            continue
        if rhok <= mu1:
            delta = delta / 4
        elif rhok > mu2 and norm(dk) == delta:
            delta = min(2 * delta, max_delta)
        if rhok > eta:
            xk = xk_1
        k += 1

        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^15.4f}{3:^15.6f}".format(
                k, norm_gk, fk, end - begin))

    print("共迭代{0}次,最优解为{1:.4f}".format(k, fk))
    return xk


def cauchy_point(gk, Bk, delta=100):
    '''cauchy point method

    Args:
        gk: the gradient of f at current iter point 'xk'
        Bk: the hessian matrix of f at current iter point 'xk'
        delta: current trust region (default: {5})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    norm_gk = norm(gk)
    ps = -delta / norm_gk * gk
    gBg = gk.T.dot(Bk.dot(gk))
    if gBg <= 0:
        pc = ps
    else:
        pc = min(1, norm_gk**3 / (delta * gBg)) * ps
    return pc


def dogleg(f, gk, Bk, delta=5, cg_exact=False, cg_iter=5, grad=None):
    '''dogleg method

    Args:
        f: object function
        gk: the gradient of object function
        Bk: the hessian matrix of object function at 'xk'
        delta: current iter trust region (default: {5})
        cg_exact: bool value to decide to use exact linear search
             (default: {True})
        cg_iter: the max iter times for CG method (default: {5})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    pu = -(gk.T.dot(gk)) / (gk.T.dot(Bk.dot(gk))) * gk
    pb = CG_modification(Bk, gk, maxiter=cg_iter,
                         exact=cg_exact, f=f, grad=grad)
    # pb = -inv(Bk).dot(gk)

    norm_pu = norm(pu)
    norm_pb = norm(pb)
    if norm_pb <= delta:
        dk = pb
    elif norm_pu >= delta:
        dk = delta / norm_pu * pu
    else:
        pbu = pb - pu
        norm_pbu_2 = norm(pbu)**2
        pu_pbu = pu.T.dot(pbu)
        tao = (-pu_pbu + sqrt((pu_pbu)**2 - norm_pbu_2 *
                              (norm(pu)**2 - delta**2))) / norm_pbu_2
        dk = pu + tao * pbu
    return dk


def LM(F, Jacobian, xk, disp=False, maxiter=n, maxiter_linear=None):
    ''' Levenberg-Marquardt 方法求解非线性最小二乘问题

    克服要求 F(x) 的 Jacobian 矩阵列满秩的缺点

    Args:
        F: object function
        Jacobian: function to get Jacobian matrix of F
        xk: current iter point 'xk'
        disp: bool value to decide to print iter info (default: {False})
        maxiter: the max iter times (default: {n})
        maxiter_linear: the max iter times for linear search (default: {None})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def norm_F(xk):
        '''返回最小二乘目标函数在 xk 处的值

        Args:
            xk: current iter point 'xk'

        Returns:
            value = 0.5 * | | F(xk) | |
            number
        '''
        return 0.5 * norm(F(xk))**2

    if disp:
        column_name(0)
        begin = time.clock()

    cmp = None
    k = 0
    eps = 1e-9
    n = xk.shape[0]
    muk = norm(F(xk))
    Ik = eye(n)

    while k < maxiter:
        fk = F(xk)
        norm_fk = norm(fk)

        if not (cmp is None):
            if abs(cmp - norm_fk) < eps:
                break
        cmp = norm_fk

        jfk = Jacobian(xk)
        gk = jfk.T.dot(fk)
        # norm_gk = norm(gk)
        if norm_fk < eps:
            break

        hessian = jfk.T.dot(jfk) + muk * Ik
        dk = CG_modification(hessian, gk, exact=True)
        # dk = np.linalg.inv(jfk.T.dot(jfk) + muk * E).dot(-gk)

        alpha = armijio(norm_F, gk, xk, dk)
        xk = xk + alpha * dk
        muk = norm(F(xk))
        k += 1
        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^15.4f}{4:^15.6f}".format(
                k, alpha, norm(gk), norm_fk, end - begin))

    print("共迭代{0}次,最优解为{1:.4f}".format(k, norm_fk))
    return xk


def gauss_newton(F, Jacobian, xk, disp=False,
                 maxiter=n, maxiter_linear=None):
    '''Gauss - Newton 法求解非线性最小二乘问题

    向量函数 F(x) 的 Jacobian 矩阵列满秩可以保证 Gauss - Newton 方向是下降方向

    Args:
        F: object function
        Jacobian: Jacobian matrix of F
        xk: current iter point 'xk'
        disp: bool value to decide to print iter info(default: {False})
        maxiter: the max iter times(default: {n})
        maxiter_linear: the max iter times for linear search(default: {None})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def norm_F(xk):
        '''返回最小二乘目标函数在 xk 处的值

        Args:
            xk: current iter point 'xk'

        Returns:
            value = 0.5 * | | F(xk) | |
            number
        '''
        return 0.5 * norm(F(xk))**2

    if disp:
        column_name(0)
        begin = time.clock()

    cmp = None
    k = 0
    eps = 1e-9
    while k < maxiter:

        rk = F(xk)
        norm_rk = norm(rk)

        if not (cmp is None):
            if abs(cmp - norm_rk) < eps:
                break
        cmp = norm_rk

        if norm_rk < eps:
            break
        Jk = Jacobian(xk)

        gk = Jk.T.dot(rk)
        hessian = Jk.T.dot(Jk)
        try:
            dk = CG_modification(hessian, gk, exact=True)
        except Exception:
            print("CG")
            return
        try:
            alpha = armijio(norm_F, gk, xk, dk, maxiter=20)
        except Exception:
            print("armijio")
            return
        xk = xk + alpha * dk
        k += 1
        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^15.4f}{4:^15.6f}".format(
                k, alpha, norm(gk), norm_rk, end - begin))

    print("共迭代{0}次,最优解为{1:.4f}".format(k, norm(rk)))
    return xk


def active_set(G, c, x0, Ae=None, be=None, Ag=None, bg=None,
               maxiter=100, disp=False):
    '''Active Set Method

    用于求解二次规划问题 min f(x) = 0.5*x'Gx + c'x
                    s.t. Aex = be , Agx >= bg

    Args:
        G: quadratic matrix
        c: coefficient vector
        x0: initial iter feasible point 'x0'
        Ae: equality matrix(default: {None})
        be: col vector for equality(default: {None})
        Ag: inequality matrix(default: {None})
        bg: col vector for inequality matrix(default: {None})
        maxiter: the max iter times(default: {100})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def fvalue(xk):
        '''返回 xk 处二次规划函数值

        Args:
            xk: current iter point 'xk'

        Returns:
            number
        '''
        return 0.5 * xk.T.dot(G.dot(xk)) + c.T.dot(xk)

    if disp:
        begin = time.clock()
    k = 0

    m, n = G.shape  # 要求 m<<n,即 G 行满秩
    if not (be is None):
        ne = be.shape[0]
    else:
        ne = 0
    xk = x0

    G_inv = inv(G)

    if not (Ag is None):
        ng = bg.shape[0]
        index = ones((ng, 1))

        for i in range(ng):
            if Ag[i].dot(xk) > bg[i]:
                index[i] = 0

        while k < maxiter:
            if not (be is None):
                A = Ae
                cmp = 1
            else:
                A = []
                cmp = 0

            for i in range(ng):
                if index[i] > 0:
                    if cmp:
                        A = vstack((A, Ag[i]))
                    else:
                        A.append(list(Ag[i]))

            if cmp == 0:
                A = array(A)
            b = zeros((A.shape[0], 1))

            gk = G.dot(xk) + c
            lamda = None
            if b.shape[0]:
                dk, lamda = null_space(G, gk, A, b)
            else:
                dk = null_space(G, gk)
            if norm(dk) > eps:
                alpha = 1.0
                tm = 1.0
                beta = 0
                idx = 0
                for i in range(ng):
                    beta = Ag[i].dot(dk)
                    if index[i] == 0 and beta < 0:
                        tm_ = (bg[i] - Ag[i].dot(xk)) / beta
                        if tm_ < tm:
                            tm = tm_
                            idx = i

                alpha = min(tm, alpha)
                xk = xk + alpha * dk

                if alpha < 1:
                    index[idx] = 1

            else:
                if not (lamda is None):
                    AH_inv = A.dot(G_inv)
                    Bk = inv(AH_inv.dot(A.T)).dot(AH_inv)
                    lamda = Bk.dot(gk)
                    min_lamda = min(lamda)
                    idx = argmin(min_lamda)
                else:
                    min_lamda = 0
                if min_lamda >= -eps:
                    break
                else:
                    for i in range(ng):
                        if index[i] and (ne + sum(index[0:i])) == idx:
                            index[i] = 0
                            break
            k += 1
    else:
        xk, lamda = null_space(G, c, Ae, be)

    if disp:
        end = time.clock()
        print("itertimes", k)
        print("花费时间：{0:.4f}".format(end - begin))
        print(lamda)
    return xk


def null_space(G, c, A=None, b=None, maxiter=n, disp=False):
    '''
    使用零空间法求解等式约束的二次规划问题
    f = 0.5 * x'Gx + c'x  s.t. Ax = b

    限制： 要求矩阵 A 行满秩（m << n）
    注意： 由于转换成以 A 零空间的基矩阵表示，优化变量变为 n - m 维

    Args:
        G: 二次项系数矩阵, n 阶对称正定
        c: 一次项系数向量, n 维列向量
        A: 约束条件矩阵， m x n行满秩矩阵
        b: 约束条件向量， m 维列向量
        x: 求解等效问题的参数向量，n - m 维列向量

    Returns:
        最优函数解向量
        n 维列向量
    '''
    if disp:
        begin = time.clock()
    if not (A is None):
        x0, Z, lamda = depack(A, b)

        W = Z.T.dot(G.dot(Z))
        cz = Z.T.dot(c + G.dot(x0))
        d = CG_modification(W, cz, maxiter=maxiter, eps=1e-9)
        # d = solve_subpro(W, cz, x)
        xpos = x0 + Z.dot(d)
        lamda = lamda.T.dot(G.dot(xpos) + c)

        if disp:
            end = time.clock()
            # print("原问题的最优解为:\n")
            # print(xpos)
            # print("对应的乘子向量为:\n")
            # print(lamda)
            print("花费时间为：\n")
            print("{0:.6f}".format(end - begin))
        return (xpos, lamda)
    else:
        return CG_modification(G, c)


def depack(A, b):
    ''' 对矩阵 A 进行 QR 分解

    Args:
        A: m X n matrix
        b: Ax = b
    '''
    m, n = A.shape
    q, r = qr(A.T)
    q1 = q[:, 0:m]
    q2 = q[:, m:]
    r = r[0:m, :]
    lamda = q1.dot(inv(r).T)
    x0 = lamda.dot(b)
    z = q2
    return (x0, z, lamda)


def lagrange(H, c, A, b, disp=False):
    '''lagrange 乘子法

    求解 f = 0.5 * x'Hx + c'x  s.t. Ax = b
    此方法是直接求 lagrange 矩阵的逆, 得到最优解

    Args:
        H: quadratic matrix
        c: coefficient vector
        A: equality matrix
        b: col vector for equality matrix
    '''
    if disp:
        begin = time.clock()
    H_inv = inv(H)
    AH_inv = A.dot(H_inv)
    AHA_inv = inv(AH_inv.dot(A.T))
    G = H_inv - AH_inv.T.dot(AHA_inv).dot(AH_inv)
    B = AHA_inv.dot(AH_inv)
    C = -AHA_inv

    x = -G.dot(c) + B.T.dot(b)
    lamda = B.dot(c) - C.dot(b)
    end = time.clock()

    if disp:
        print("原问题的最优解为:\n")
        print(x)
        print("对应的乘子向量为:\n")
        print(lamda)
        print("花费时间：\n")
        print("{0:.6f}".format(end - begin))

    return (x, lamda)


def feasible_direction(f, grad, xk, Ag, bg, Ae=None, be=None,
                       eps=1e-5, maxiter=200, disp=False):
    '''可行方向法求解线性约束问题
    在求解线性规划子问题时采用单纯形法的精度会高一些，但是由于目前还没有写出
    比较完备的单纯形代码，所以用的是罚函数法中的外点法(这里因为内点法对于初始
    可行点的要求在一般问题中难以达到，且增广拉格朗日乘子法处理线性规划有点
    小题大做的意味，所以选择效果居中，操作简便的外点法)，但是这样得出的解的
    误差较大，该函数还有点优化

    min f(x) s.t. Ax >= b , Ex = e ( x_i > 0 )

    Args:
        f: object function
        grad: the gradient of object function
        xk: initial iter point 'xk'
        Ag: inequality constraints matrix
        bg: inequality constraints col vector
        Ae: equality constraints matrix (default: {None})
        be: equality constraints col vector (default: {None})
        eps: Allowable error range (default: {1e-5})
        maxiter: the max iter times (default: {200})
        disp: bool value to decide to print iter info or not
             (default: {False})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def LP(dk, c):
        return c.T.dot(dk)

    k = 0
    m, n = Ag.shape

    while k < maxiter:
        cmp = 0
        index = zeros((m, 1))

        A = Ag.dot(xk)
        for i in range(m):
            if A[i] == bg[i]:
                index[i] = 1
                cmp = 1
        gk = grad(xk)

        if not (be is None) or cmp:
            if not (be is None):
                E = Ae
                e = zeros((Ae.shape[0], 1))
            else:
                E = None
                e = None
            if cmp:
                A1 = []
                b1 = []
                for row in range(m):
                    if index[row]:
                        A1.append(Ag[row])
                        b1.append(bg[row, 0])
                # temp_eye = eye(n, n)
                # temp_ones = ones((n, 1))
                A1 = array(A1)
                b1 = array(b1).reshape((-1, 1))
                # A1 = vstack((A1, temp_eye))
                # b1 = vstack((b1, -1 * temp_ones))
                # A1 = vstack((A1, -1 * temp_eye))
                # b1 = vstack((b1, -1 * temp_ones))
            else:
                A1 = None
                b1 = None
            objfunc = partial(LP, c=gk)
            dk_start = 0.1 * ones((n, 1))
            dk = exterior_point_matrix(
                objfunc, dk_start, Ae=E, be=e, Ag=A1, bg=b1)

            zk = gk.T.dot(dk)
            if abs(zk) < eps:
                break

        else:
            if norm(gk) < eps:
                break
            dk = -gk

        alpha_list = []
        for i in range(m):
            if index[i] == 0:
                Ad = Ag[i].dot(dk)
                if Ad < 0:
                    alpha_list.append(float((bg[i] - A[i]) / Ad))

        if len(alpha_list):
            alpha = min(alpha_list)
            alpha = direct_search(
                (lambda s: f(xk + s * dk)), 0, alpha, e=1e-5, maxiter=25)
        else:
            left, right = search_interval(
                (lambda s: f(xk + s * dk)), 0, 1e-4, 1.1, itr=25)
            alpha = direct_search(
                (lambda s: f(xk + s * dk)), left, right, e=1e-5, maxiter=25)

        xk = xk + alpha * dk
        k += 1
    return xk


def rosen(f, grad, xk, Ag, bg,
          E=None, e=None, maxiter=25, eps=1e-10, disp=False, linear_flag=0):
    '''梯度投影算法

    主要用于求解线性约束问题 min f(x) s.t. Ax >= b , Ex = e

    Args:
        f: object function
        grad: the gradient of object function
        xk: initial point 'xk'
        Ag: inequality constraints matrix
        bg: inequality constraints col vector
        E: equality constraints matrix(default: {None})
        e: equality constraints col vector  (default: {None})
        maxiter: the max iter times (default: {1000})
        eps: Allowable error range (default: {1e-5})
        disp: bool value to decide to print iter info (default: {False})
        linear_flag: decide to use interpolation or direct_search
            (default: {0})  < 0:q_interpolation, 1:direct_search >

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    k = 0
    m = Ag.shape[0]
    n = xk.shape[0]
    length_E = 0
    xk_former = xk

    if disp:
        begin = time.clock()
        column_name(1)

    while k < maxiter:
        if not (xk_former is xk) or k == 0:
            M = []
            index = zeros((m, 1))
            for i in range(m):
                if Ag[i].dot(xk) == bg[i]:
                    index[i] = 1
                    M.append(list(Ag[i]))

            M = array(M)
            if not (E is None):
                length_E = E.shape[0]
                if M.shape[0]:
                    M = vstack((M, E))
                else:
                    M = E.copy()

        length_A1 = M.shape[0] - length_E
        if M.shape[0] == 0:
            P = eye(n, n)
        else:
            p = inv(M.dot(M.T)).dot(M)
            P = eye(n) - M.T.dot(p)

        gk = grad(xk)
        dk = -P.dot(gk)
        if norm(dk) < eps:
            w = (inv(M.dot(M.T)).dot(M)).dot(gk)
            lamda = w[0:length_A1]
            cmp = True
            for i in range(length_A1):
                if lamda[i] < 0:
                    M = delete(M, i, 0)
                    cmp = False
                    break
            if cmp:
                break
            else:
                xk_former = xk

        else:
            alpha_list = []
            for i in range(m):
                if index[i] == 0:
                    t = Ag[i].dot(dk)
                    if t < 0:
                        alpha_list.append(float((bg[i] - Ag[i].dot(xk)) / t))
            if len(alpha_list):
                alpha_ba = min(alpha_list)
            else:
                alpha_ba = 1

            if linear_flag == 0:
                alpha = q_interpolation(
                    (lambda alpha: f(xk + alpha * dk)), 0, alpha_ba,
                    maxiter=25)
                alpha = alpha_ba if alpha > alpha_ba else alpha
            else:
                alpha = direct_search(
                    (lambda alpha: f(xk + alpha * dk)), 0, alpha_ba, eps,
                    maxiter=25)

            xk_former = xk
            xk = xk + alpha * dk
            if norm(xk - xk_former) < eps:
                break

        k += 1
        if disp:
            end = time.clock()
            print("{0:<6d}{1:^10.5f}{2:^15.5f}{3:^15.6f}".format(
                k, norm(gk), f(xk), end - begin))

    if disp:
        print("最优解是：")
        print(xk)
        print("最优解处的函数值是：")
        print(f(xk))

    return xk


def exterior_point_matrix(f, xk, Ae=None, be=None, Ag=None, bg=None,
                          mu=None, ita=8, maxiter=25, eps=1e-5, disp=False):
    '''外点法求解线性矩阵约束问题的版本

    min f(x) s.t. Aex=be , Agx >= bg (x_i >= 0)

    Args:
        f: object function
        xk: initial iter point 'xk'
        Ae: equality constraints matrix (default: {None})
        be: equality constraints col vector (default: {None})
        Ag: inequality constraints matrix (default: {None})
        bg: inequality constraints col vector (default: {None})
        mu: penalty parameters (default: {None})
        ita: args to change penalty parameters (default: {0.45})
        maxiter: the max iter times (default: {25})
        eps: Allowable error range (default: {1e-5})
        disp: bool value to decide to print iter info or not
             (default: {False})

    Returns:
        [description] 用一般约束问题版本函数处理
        [type] funciton
    '''
    def constraints_func(xk, A, b):
        return A.dot(xk) - b

    if not (Ae is None):
        equality = partial(constraints_func, A=Ae, b=be)
    else:
        equality = None
    if not (Ag is None):
        greater = partial(constraints_func, A=Ag, b=bg)
    else:
        greater = None
    return exterior_point(f, xk, h=equality, g=greater,
                          mu=mu, ita=ita, maxiter=maxiter, eps=eps, disp=disp)


def exterior_point(f, xk, h=None, g=None,
                   mu=None, ita=8, maxiter=25, eps=1e-5, disp=False):
    '''外点法求解一般约束优化问题
    外点法相对于内点法而言，对初始迭代点没有要求，且在处理小规模问题时，近似解的
    误差还算是能够接受的

    min f(xk) s.t. h(x) = 0 , g(x) >= 0

    Args:
        f: object function
        xk: initial iter point 'xk'
        h: equality constraints (default: {None})
        g: inequality constraints (default: {None})
        mu: penalty parameters (default: {None})
        ita: args to change penalty parameters (default: {0.45})
        maxiter: the max iter times (default: {25})
        eps: Allowable error range  (default: {1e-5})
        disp: bool value to decide to print iter info or not
            (default: {False})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def penalty(xk, f, mu, h=None, g=None):
        '''构建外点法的无约束最优化模型

        P(x) = f(x) + mu * [ sum h_i(x)**2 +  sum min(0 , g_i(x))**2 ]

        Args:
            xk: current iter point 'xk'
            f: object function
            mu: penalty parameter
            h: equality constraints (default: {None})
            g: inequality constraints (default: {None})

        Returns:
            [description] 返回目标函数对应约束条件的罚函数模型在xk处的值
            [type] number
        '''
        fk = f(xk)
        sum_h = 0
        sum_g = 0
        if not (h is None):
            hk = h(xk)
            mh = hk.shape[0]
            for row in range(mh):
                sum_h += hk[row, 0]**2
        if not (g is None):
            gk = g(xk)
            mg = gk.shape[0]
            for row in range(mg):
                sum_g += min(0, gk[row, 0])**2
        return fk + mu * (sum_h + sum_g)

    k = 0
    n = xk.shape[0]

    if disp:
        begin = time.clock()
        print("{0:<6s}{1:^15s}{2:^15s}".format(
            "iter", "fvalue", "time"))

    if mu is None:
        mu = 0.1
    while k < maxiter:
        minimize_obj = partial(penalty, f=f, mu=mu, h=h, g=g)
        xk = powell(minimize_obj, xk, maxiter=15 * n)
        mu *= ita
        if norm(minimize_obj(xk) - f(xk)) < eps:
            break
        k += 1

        if disp:
            end = time.clock()
            print("{0:<6d}{1:^15.4f}{2:^15.6f}".format(
                k, f(xk), end - begin))
    if disp:
        print("最优解为：", f(xk))
        print("最优点为：")
        print(xk)
    return xk


def interior_point_matrix(f, xk, Ae=None, be=None, Ag=None, bg=None,
                          mu=None, ita=0.45,
                          maxiter=25, eps=1e-5, disp=False):
    '''内点法求解线性矩阵约束问题的版本

    min f(x) s.t. Aex=be , Agx >= bg (x_i >= 0)

    Args:
        f: object function
        xk: initial iter point 'xk'
        Ae: equality constraints matrix (default: {None})
        be: equality constraints col vector (default: {None})
        Ag: inequality constraints matrix (default: {None})
        bg: inequality constraints col vector (default: {None})
        mu: penalty parameters (default: {None})
        ita: args to change penalty parameters (default: {0.45})
        maxiter: the max iter times (default: {25})
        eps: Allowable error range (default: {1e-5})
        disp: bool value to decide to print iter info or not
             (default: {False})

    Returns:
        [description] 用一般约束问题版本函数来处理
        [type] function
    '''
    def constraints_func(xk, A, b):
        return A.dot(xk) - b

    if not (Ae is None):
        equality = partial(constraints_func, A=Ae, b=be)
    else:
        equality = None
    if not (Ag is None):
        greater = partial(constraints_func, A=Ag, b=bg)
    else:
        greater = None
    return interior_point(f, xk, h=equality, g=greater,
                          mu=mu, ita=ita, maxiter=maxiter, eps=eps, disp=disp)


def interior_point(f, xk, h=None, g=None,
                   mu=None, ita=0.45, maxiter=25, eps=1e-5, disp=False):
    '''内点法求解一般约束优化问题
    内点法的致命缺点是严格初始可行点的选择，且内点法对于参数十分敏感，
    经常得不到理想精度下的解

    min f(xk) s.t. h(x) = 0 , g(x) >= 0

    Args:
        f: object function
        xk: initial iter point 'xk'
        h: equality constraints (default: {None})
        g: inequality constraints (default: {None})
        mu: penalty parameters (default: {None})
        ita: args to change penalty parameters (default: {0.45})
        maxiter: the max iter times (default: {25})
        eps: Allowable error range  (default: {1e-5})
        disp: bool value to decide to print iter info or not
            (default: {False})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def penalty(xk, f, mu, h=None, g=None):
        '''构建内点法的无约束最优化模型

        P(x) = f(x) + 1/(2*mu) * sum h_i(x) + mu * sum [ 1. / g_i(x) ]

        Args:
            xk: current iter point 'xk'
            f: object function
            mu: penalty parameter
            h: equality constraints (default: {None})
            g: inequality constraints (default: {None})

        Returns:
            [description] 返回目标函数对应约束条件的罚函数模型在xk处的值
            [type] number
        '''
        fk = f(xk)
        sum_h = 0
        sum_g = 0
        if not (h is None):
            hk = h(xk)
            mh = hk.shape[0]
            for row in range(mh):
                sum_h += hk[row, 0]**2
        if not (g is None):
            gk = g(xk)
            mg = gk.shape[0]
            for row in range(mg):
                if gk[row, 0]:
                    sum_g += 1. / gk[row, 0]
                else:
                    sum_g += 1e5
        return fk + 1. / (2 * mu) * sum_h + mu * sum_g

    def cal_mu(f_value, g_value):
        '''动态初始罚参数值

        mu = f(xk) / [1 / sum g_i(xk)]

        Args:
            f_value: value of object function at 'xk'
            g_value: value of inequality constraints at 'xk'

        Returns:
            [description] 返回初始罚参数值
            [type] float
        '''
        if f_value:
            m = g_value.shape[0]
            sum_g = 0
            for row in range(m):
                if g_value[row, 0]:
                    sum_g += 1. / g_value[row, 0]
                else:
                    sum_g += 1e5
            if sum_g:
                mu = abs(f_value / sum_g)
            else:
                mu = 50
        else:
            mu = 50
        return mu

    k = 0

    if disp:
        begin = time.clock()
        print("{0:<6s}{1:^15s}{2:^15s}".format(
            "iter", "fvalue", "time"))
    n = xk.shape[0]
    if mu is None:
        fk = f(xk)
        gk = g(xk)
        mu = cal_mu(fk, gk)
    while k < maxiter:

        minimize_obj = partial(penalty, f=f, h=h, g=g, mu=mu)
        xk = powell(minimize_obj, xk, maxiter=20 * n)
        mu *= ita

        if minimize_obj(xk) - f(xk) < eps or mu < eps:
            break
        k += 1
        if disp:
            end = time.clock()
            print("{0:<6d}{1:^15.4f}{2:^15.6f}".format(
                k, f(xk), end - begin))
    if disp:
        print("最优解为：", f(xk))
        print("最优点为：")
        print(xk)
    return xk


def augmented_lagrange(xk, f, h=None, g=None, mu=None,
                       lamda=None, sigma=None):
    '''构建增广拉格朗日函数

    如果得定等式约束 h，那么一定要给定参数 mu
    如果给定不等式约束 g，那么一定要给定参数 lamda

    Args:
        xk: current iter point 'xk'
        f: object function
        h: equality constraints (default: {None})
        g: inequality constraints (default: {None})
        mu: multiplier vector for equality constraints (default: {None})
        lamda: multiplier vector for revised inequality constraints
            (default: {None})
        sigma: penalty parameters (default: {None})

    Returns:
        [description]augmented_lagrange function at 'xk'
        [type] float
    '''
    fk = f(xk)
    if not(h is None):
        hk = h(xk)
        sum_hi_2 = 0.5 * sigma * norm(hk)**2
        sum_hi = mu.T.dot(hk)
    else:
        sum_hi_2 = 0
        sum_hi = 0

    if not(g is None):
        gk = g(xk)
        sum_gi = 0
        m = gk.shape[0]

        for i in range(m):
            sum_gi += min(0, float(sigma * gk[i] - lamda[i]))**2 - lamda[i]**2
        sum_gi = 0.5 * sum_gi / sigma
    else:
        sum_gi = 0

    return float(fk - sum_hi + sum_hi_2 + sum_gi)


def multiplier_method_matrix(f, xk, Ae=None, be=None, Ag=None, bg=None,
                             mu=None, lamda=None, sigma=None,
                             maxiter=25, eps=1e-5,
                             ita=0.5, aita=2.0, disp=False):
    '''增广拉格朗日方法对于求解线性矩阵约束的版本

    Args:
        f: object function
        xk: initial iter point 'xk'
        Ae: equality constraints matrix (default: {None})
        be: equality constraints col vector (default: {None})
        Ag: inequality constraints matrix (default: {None})
        bg: inequality constraints col vector (default: {None})
        mu: multiplier vector for equlity constraints (default: {None})
        lamda: multiplier vector for inequality constraints (default: {None})
        sigma: penalty parameters (default: {None})
        maxiter: the max iter times (default: {25})
        eps: Allowable error range (default: {1e-5})
        ita: args to decide to change sigma or not (default: {0.8})
        aita: args for changing sigma (default: {2.0})
        disp: bool value to decide to print iter info (default: {False})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def constraints_func(xk, A, b):
        return A.dot(xk) - b

    if not (Ae is None):
        equality = partial(constraints_func, A=Ae, b=be)
    else:
        equality = None
    if not (Ag is None):
        greater = partial(constraints_func, A=Ag, b=bg)
    else:
        greater = None
    return multiplier_method(f, xk, h=equality, g=greater,
                             mu=mu, lamda=lamda, sigma=sigma,
                             maxiter=maxiter, eps=eps,
                             ita=ita, aita=aita, disp=disp)


def multiplier_method(f, xk, h=None, g=None, mu=None, lamda=None, sigma=None,
                      maxiter=25, eps=1e-5, ita=0.45, aita=1.1, disp=False):
    '''使用增广拉格朗日算法求解非线性约束问题

    主要用于求解 min f(x) s.t. h(x) = 0 , g(x) >= 0
    增广拉格朗日函数为 G(x) = f(x) - Sum mu_i*h_i(x) +
        0.5*sigma * Sum h_i(x)^2 + (1/2*sigma)*Sum (min{0,
        sigma*g_i(x)-lamda_i}^2-lamda_i^2)

    Args:
        f: object function
        h: equality constraints(vector valued  function)
        g: inequality constraints(vector valued  function)
        xk: initial iter point 'xk'
        mu: multiplier vector for equality contraints  (default: {None})
        lamda: multiplier vector for revised inequality contraints
         (default: {None})
        sigma: penalty parameters (default: {None})
        maxiter: the max iter times (default: {1000})
        eps: Allowable error range (default: {1e-5})
        ita: args to decide to change sigma or not (default: {0.8})
        aita: args for changing sigma (default: {2.0})
        disp: bool value to decide to print iter info (default: {False})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    k = 0
    beta_ = 0
    n = xk.shape[0]
    if not(h is None):
        hk = h(xk)
        m = hk.shape[0]

    if not(g is None):
        gk = g(xk)
        s = gk.shape[0]

    if disp:
        begin = time.clock()
        print("{0:<6s}{1:^15s}{2:^15s}".format(
            "iter", "fvalue", "time"))

    if mu is None and not(h is None):
        mu = 0.1 * ones((m, 1))
    if lamda is None and not(g is None):
        lamda = 0.1 * ones((s, 1))
    if sigma is None:
        sigma = 1.1

    while k < maxiter:
        penalty = partial(augmented_lagrange, f=f, h=h, g=g,
                          mu=mu, lamda=lamda, sigma=sigma)
        if not(g is None):
            xk = powell(penalty, xk, maxiter=20 * n)
            if not(h is None):
                hk = h(xk)
                beta = sum(hk**2)
            else:
                beta = 0

            gk = g(xk)
            m = gk.shape[0]
            for i in range(m):
                beta += min(gk[i], float(lamda[i] / sigma))**2
            beta = sqrt(beta)
            if beta < eps:
                break
            else:
                if beta > ita * beta_:
                    sigma = aita * sigma
                if not(h is None):
                    mu = mu - sigma * hk
                for i in range(m):
                    lamda[i] = max(0, lamda[i] - gk[i])

            beta_ = beta
        elif not(h is None) and g is None:
            hk = h(xk)
            norm_hk = norm(hk)
            xk = powell(penalty, xk, maxiter=20 * n)
            hk_1 = h(xk)
            norm_hk_1 = norm(hk_1)
            if norm_hk_1 < eps:
                break
            else:
                if norm_hk_1 >= ita * norm_hk:
                    sigma = aita * sigma
                mu = mu - sigma * hk_1

        else:
            xk = powell(f, xk, disp=disp)
            break

        k += 1

        if disp:
            fk = f(xk)
            end = time.clock()
            print("{0:<6d}{1:^15.4f}{2:^15.6f}".format(
                k, fk, end - begin))

    if disp:
        print("最优解为：")
        print(xk)
        print("最优值为：")
        print(f(xk))
    return (xk, mu, lamda)


def newton_lagrange(f, h, df, dh, d2f, d2h, xk, mu, maxiter=1000,
                    eps=1e-5, rho=0.5, gamma=0.4, disp=False):
    '''基于牛顿下降的lagrange算法求解一般等式约束问题
    此算法的致命缺点是要不断地计算梯度与 hesse 阵,
    首先是梯度与 hesse 阵的计算是否容易，其次是要保证 hesse 阵正定
    如果采用 CG 算法求解牛顿问题，保证精确但耗时严重，不精确但结果误差大

    min f(x) s.t. h(x) = 0

    Args:
        f: object function
        h: equality constraints
        df: the gradients of object function
        dh: the gradients of equality constraints
        d2f: the hesse matrix of object function
        d2h: the hesse matrix of equality constrains
        xk: initial iter point 'xk'
        mu: multiplier vector for equality constraints
        maxiter: the max iter times (default: {1000})
        eps: Allowable error range (default: {1e-5})
        rho: args for end (default: {0.5})
        gamma: args for end [description] (default: {0.4})
        disp: bool value to decide to print iter info or not
             (default: {False})

    Returns:
        [description] shift of current iter
        [type] col vector
    '''
    def f_lagrange(xk, mu):
        '''lagrange 函数

        L(x,mu) = f(x) - mu.T.dot(h(x))

        Args:
            xk: current iter point 'xk'
            mu: multiplier vector

        Returns:
            [description] 返回 lagrange 函数在 xk 处的值
            [type] number
        '''
        return float(f(xk) - mu.T.dot(h(xk)))

    def df_lagrange(xk, mu):
        '''返回 lagrange 函数的梯度向量

        Args:
            xk: current iter point 'xk'
            mu: multiplier vector

        Returns:
            [description] 返回 lagrange 函数在当前点 xk 处的梯度
            [type] col vector
        '''
        return vstack((df(xk) - dh(xk).dot(mu), -h(xk)))

    def d2f_lagrange(xk, mu):
        '''返回 lagrange 函数的 hesse matrix

        Args:
            xk: current iter point 'xk'
            mu: multiplier vector

        Returns:
            [description] 返回 lagrange 函数在当前点 xk 处的 hesse 矩阵
            [type] matrix
        '''
        d2h_matrix = d2h(xk)
        m = len(d2h_matrix)
        dw = d2f(xk)
        dh_matrix = dh(xk)
        for i in range(m):
            dw = dw - float(mu[i]) * d2h_matrix[i]

        return vstack((hstack((dw, -dh_matrix)), hstack((-dh_matrix.T, zeros((
            dh_matrix.shape[1], dh_matrix.shape[1]))))))

    k = 0
    m = xk.shape[0]

    while k < maxiter:
        dl_matrix = df_lagrange(xk, mu)
        norm_dl_matrix = norm(dl_matrix)
        if norm_dl_matrix < eps:
            break
        d2l_matrix = d2f_lagrange(xk, mu)
        # dk_vk = CG_modification(d2l_matrix, dl_matrix, maxiter=10)
        dk_vk = -inv(d2l_matrix).dot(dl_matrix)
        dk = dk_vk[0:m, :]
        vk = dk_vk[m:, :]
        dl_1_maxtrix = df_lagrange(xk + dk, mu + vk)
        if norm(dl_1_maxtrix)**2 <= (1 - gamma) * norm_dl_matrix**2:
            alpha = 1
        else:
            j = 0
            p = 0
            while j < 20:
                beta = rho**p
                if norm(df_lagrange(xk + beta * dk, mu + beta * vk))**2 \
                        > (1 - gamma * beta) * norm_dl_matrix**2:
                    p += 1
                else:
                    break
                j += 1
            alpha = beta
        xk = xk + alpha * dk
        mu = mu + alpha * vk
        k += 1
    return xk


def Simplex(c, Ae, be):

    m, n = Ae.shape
    S = hstack((c.T, array([[0]])))
    S = vstack((S, hstack((Ae, be))))
    for i in range(1, m + 1):
        S[i] = S[i] / S[i, i - 1]
        for j in range(i + 1, m + 1):
            S[j] = S[j] - S[i] * S[j, i - 1]
    for i in range(m - 1):
        for j in range(m - i - 1):
            S[m - i - j - 1] = S[m - i - j - 1] - \
                S[m - i - j - 1, m - i - 1] * S[m - i]
    for i in range(m):
        S[0] = S[0] - S[0, i] * S[i + 1]

    d_check = (S[0, m], m)
    for i in range(n - m):
        d_check = (S[0, i + m], m + i) if S[0, i + m] > d_check[0] else d_check

    if d_check[0] <= 0:
        return S[:, -1]
    else:
        A = S[:, d_check[1]]
        t = S.shape[1]
        r = array([max(float(be[i] / A[i]), 0) for i in range(t)])
        r_min = min(r)
        idx = argmin(r)
        S[idx, :] = S[idx, :] / r_min
