# -*- coding:utf-8 -*-

'''测试 minimize中的优化函数

   @Author     : garvenlee
   @DateTime   : 2019-01-04 17:23:58
   @Version    : python 3.6.5

'''

from minimize import *
from numpy import random, eye, array, zeros, exp, sin, cos


def objfunc(x):
    x = array(x)
    return float(0.5 * x.T.dot(A.dot(x)) + c.T.dot(x))


def grad(x):
    return A.dot(x) + c


def hesse(x):
    return A


def leastsq(x):
    '''预测值函数y = 5 + 4t - 3t^2 + 2e^(-0.5t)
        模拟函数f = x1 + x2*t - x3*t^2 + x4*e^(x5*t)
        观测时间t = 0,1,2,...,9共九个时间点，得到相应的观测值，	用模拟函数最小化norm(F),F是一个fi-yi的向量值函数
    '''
    f = []
    for i in range(10):
        t = i + 1
        y = 5 + 4 * t - 3 * t**2 + 2 * \
            exp(-t * 0.5)
        f.append(x[0] + t * x[1] + t**2 * x[2] +
                 x[3] * exp(-t * x[4]) - y)
    res = array(f).reshape((10, -1))
    return res


def jacobian(x):
    '''返回向量值函数的Jacobian矩阵，要求列满秩
    '''
    j = []
    for i in range(10):
        t = i + 1
        temp = exp(-t * x[4])
        fpartial = [1, t, t**2, temp, x[3] * (-t) * temp]
        j.append(fpartial)
    return array(j).reshape((10, -1))


def fsolve(x):
    y1 = x[0] - 0.7 * sin(x[0]) - 0.2 * cos(x[1])
    y2 = x[1] - 0.7 * cos(x[0]) + 0.2 * sin(x[1])
    y = array([y1, y2]).reshape((2, -1))
    return y


def jsolve(x):
    jac = []
    jac.append([1 - 0.7 * cos(x[0]), 0.2 * sin(x[1])])
    jac.append([0.7 * sin(x[0]), 1 + 0.2 * cos(x[1])])
    return array(jac).reshape((2, -1))


if __name__ == "__main__":
    # ======================== 测试无约束优化
    m = 5
    n = 50
    A = random.randn(m, n)
    A = A.T.dot(A) + 0.01 * eye(n, n)
    c = random.randn(n, 1)

    xk = random.random((n, 1))

    print("最速下降法\n")
    steepest_decent(objfunc, grad, xk, maxiter_linear=20, disp=True)

    print("加速最速下降法\n")
    steepest_decent(objfunc, grad, xk, momentum_flag=True,
                    momentum=0.5, maxiter_linear=100)

    print("共轭梯度法\n")
    conjugate_gradient(objfunc, grad, hesse, xk, exact=True, disp=True)

    print("线搜索牛顿CG法\n")
    linear_newton_CG(objfunc, grad, hesse, xk, disp=True, exact=True)

    print("BFGS法\n")
    bfgs(objfunc, grad, xk, disp=True)

    print("lbfgs\n")
    lbfgs(objfunc, grad, xk, disp=True)

    print("dogleg\n")
    trust_region(objfunc, grad, hesse, xk, disp=True, method_flag=0)

    print("cauchy point\n")
    trust_region(objfunc, grad, hesse, xk, disp=True, method_flag=1)

    # =========================== 测试等式凸二次规划问题
    m = 5
    n = 1000
    G = random.randn(m, n)
    G = G.T.dot(G) + 1e-5 * eye(n, n)
    c = random.randn(n, 1)
    A = random.randn(m, n)
    b = random.randn(m, 1)
    x0 = zeros((n - m, 1))
    x, lamda = null_space(G, c, A, b, maxiter=100, disp=True)
    print("{0:.2e}".format(norm(A.dot(x) - b)))
    x, lamda = lagrange(G, c, A, b, disp=True)
    print("{0:.2e}".format(norm(A.dot(x) - b)))

    # ==================== 测试有效集方法求解 一般凸二次规划
    n = 500
    m = 30
    j = 50
    H = random.randn(m, n)
    H = H.T.dot(H) + 0.01 * eye(n, n)
    c = random.random((n, 1))
    x0 = zeros((n, 1))
    Ae = random.random((m, n))
    be = zeros((m, 1))
    Ag = random.random((m, n))
    bg = zeros((m, 1))

    print(0.5 * x0.T.dot(H.dot(x0)) + c.T.dot(x0))
    xk = active_set(H, c, x0, Ae=Ae, be=be, Ag=Ag,
                    bg=bg, disp=True, maxiter=200)
    print(0.5 * xk.T.dot(H.dot(xk)) + c.T.dot(xk))
    print(norm(Ae.dot(xk) - be))
    xk = CG_modification(H, c)
    print(0.5 * xk.T.dot(H.dot(xk)) + c.T.dot(xk))

    # # ============================== 测试非线性最小二乘
    x0 = zeros((5, 1))
    # x0 = ones((5, 1))
    # x0 = random.randn(5, 1)
    # x0 = -10 * ones((5, 1))
    xk = gauss_newton(leastsq, jacobian, x0, disp=True, maxiter=1000)
    print(xk)
    xk = LM(leastsq, jacobian, x0, disp=True, maxiter=1000)
    print(xk)

    # ============================ 借助非线性最小二乘问题思路求解方程组
    x0 = zeros((2, 1))
    xk = gauss_newton(fsolve, jsolve, x0, disp=True)
    print(xk)
    xk = LM(fsolve, jsolve, x0, disp=True)
    print(xk)
