# -*- coding:utf-8 -*-

'''
   @Author     : garvenlee
   @DateTime   : 2018-12-30 11:08:47
   @Version    : python 3.6.5

'''

import time
from numpy.linalg import (norm, eigvals)
from numpy import (arange, meshgrid, diag, array, Inf, shape, mat, delete,
                   asarray, zeros, random, eye, ones, sqrt, vstack, argmin)
import matplotlib.pyplot as plt
from scipy.linalg import inv, solve, qr

eps = 1e-5
m = 5
n = 1000
A = random.randn(m, n)
A = A.T.dot(A) + 0.01 * eye(n, n)
c = random.randn(n, 1)

_all = [
    'steepest_decent', 'conjugate_gradient', 'bfgs', 'newton_CG'
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
        print("{0:<6s}{1:^10s}{2:^10s}{3:10s}{4:10s}".format(
            "iter", "alpha", "norm_dk", "fvalue", "time"))
    else:
        print("{0:6<s}{1:^10s}{2:^10s}{3:^10s}".format(
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


def search_interval(func, start, step_h, gamma, itr=5):
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
        func, a, b, e, al=None, ar=None, lvalue=None, rvalue=None, maxiter=5):
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
    return ar if al is None else al


def steepest_decent(f, grad, xk, c1=None, c2=None, beta=None,
                    sigma=None, maxiter=3000, maxiter_linear=None,
                    exact=False, disp=True,
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
            left, right = search_interval(
                (lambda alpha: f(xk + alpha * dk)), 0.1, 0.01, 2)
            alpha = direct_search(
                (lambda alpha: f(xk + alpha * dk)), left, right, 1e-5)

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


def powell(f, xk, maxiter=5, disp=False):
    '''powell 算法不需要梯度求解凸最优化问题

    Args:
        f: object function
        xk: current iter point 'xk'
        maxiter: the max iter times (default: {5})

    Returns:
        [description] optimum solution
        [type] col vector
    '''
    k = 0
    n = xk.shape[0]
    Ik = eye(n)

    if disp:
        print("{0:<6s}{1:^15s}".format('iter', 'f_value'))

    while k < maxiter:
        former = xk
        if disp:
            fk = f(xk)
        fk = f(xk)
        xpos = []
        xpos.append(xk)
        for i in range(n):
            vk = Ik[i].reshape((n, -1))
            left, right = search_interval(
                (lambda alpha: f(xpos[i] + alpha * vk)), 0.1, 0.01, 2)
            alpha = direct_search(
                (lambda alpha: f(xpos[i] + alpha * vk)), left, right, 1e-5)
            xpos.append(xpos[i] + alpha * vk)
        delete(Ik, 0, 0)
        vstack((xpos[-1] - xpos[0]).reshape((-1, n)))
        vk = xpos[-1] - xpos[0]
        left, right = search_interval(
            (lambda alpha: f(xpos[-1] + alpha * vk)), 0.1, 0.01, 2)
        alpha = direct_search(
            (lambda alpha: f(xpos[-1] + alpha * vk)), left, right, 1e-5)
        xk = xpos[-1] + alpha * vk

        if disp:
            print('{0:<6d}{1:^15.4f}'.format(k, fk))

        if norm(xk - former) < eps:
            break

        k += 1

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
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^10.6f}".format(
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

    克服 F(x) 的 Jacobian 矩阵列满秩的缺点

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
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^10.4f}{4:^10.6f}".format(
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
            print("{0:<6d}{1:^10.4f}{2:^10.4f}{3:^10.4f}{4:^10.6f}".format(
                k, alpha, norm(gk), norm_rk, end - begin))

    print("共迭代{0}次,最优解为{1:.4f}".format(k, norm(rk)))
    return xk


def active_set(G, c, x0, Ae=None, be=None, Ag=None, bg=None,
               maxiter=100, disp=False):
    '''Active Set Method

    用于求解二次规划问题

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
        # print("原问题的最优解为:\n")
        # print(x)
        # print("对应的乘子向量为:\n")
        # print(lamda)
        print("花费时间：\n")
        print("{0:.6f}".format(end - begin))

    return (x, lamda)
