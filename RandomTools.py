################
## 包含随机矩阵的部分，其它的数值工具不常用（用 torch 代替，准备删除

import numpy as np
import scipy.linalg as la


# def deriv(f: "function", x0: "int", par=(), char_len=1, error=1e-5):
#     """df/dx(x0)
#     """
#     delta_x = char_len
#     pars = (x0 + delta_x,) + par
#     yp = f(*pars)
#     pars = (x0 - delta_x,) + par
#     ym = f(*pars)
#     lastdydx = (yp - ym) / 2 / delta_x
#     for _ in range(20):
#         delta_x = delta_x / 2
#         pars = (x0 + delta_x,) + par
#         yp = f(*pars)
#         pars = (x0 - delta_x,) + par
#         ym = f(*pars)
#         dydx = (yp - ym) / 2 / delta_x
#         if abs(dydx - lastdydx) < error:
#             break
#         lastdydx = dydx
#     return dydx


# def partial_deriv(f: "function", i: "int", x0: "list", par=(), char_len=1, error=1e-5):
#     """ ∂ᵢf(x0)  
#     par is the parameters of f
#     the derivative a evaluated with the initial length char_len
#     (x0 - 1, x0 + 1) 然后二分的计算差分 直到第一次收敛到 error 内
#     """
#     delta_x = char_len
#     x = [xi + delta_x if ind == i else xi for ind, xi in enumerate(x0)]
#     pars = (x,) + par
#     yp = f(*pars)
#     pars = (x0,) + par
#     # y0 = f(*pars)
#     # lastdydx = (yp - y0)/delta_x
#     x = [xi - delta_x if ind == i else xi for ind, xi in enumerate(x0)]
#     pars = (x,) + par
#     ym = f(*pars)
#     lastdydx = (yp - ym) / 2 / delta_x
#     for _ in range(20):
#         delta_x = delta_x / 2
#         x = [xi + delta_x if ind == i else xi for ind, xi in enumerate(x0)]
#         pars = (x,) + par
#         yp = f(*pars)
#         # dydx = (yp - y0)/delta_x
#         x = [xi - delta_x if ind == i else xi for ind, xi in enumerate(x0)]
#         pars = (x,) + par
#         ym = f(*pars)
#         dydx = (yp - ym) / 2 / delta_x
#         if abs(dydx - lastdydx) < error:
#             break
#         lastdydx = dydx
#     return dydx


# def multi_partial(f: "function", xi: "list", x0: "list", par=(), char_len=1, error=1e-5):
#     """∂ᵢ∂ⱼ...∂ₖf(x0)
#     xi: [k,...,j,i]
#     par is the parameters of f
#     the derivative a evaluated with the initial length char_len
#     (x0 - 1, x0 + 1) 然后二分的计算差分 直到第一次收敛到 error 内
#     对于大于三阶的导数运行就已经很慢了，二阶三阶的导数勉强可以用
#     """
#     if len(xi) == 1:
#         return partial_deriv(f, xi[0], x0, par, char_len, error)
#     for _ in xi:
#         dfdx = lambda x, * \
#             par: multi_partial(f, xi[1:], x, par, char_len, error * 0.1)
#         return partial_deriv(dfdx, xi[0], x0, par, char_len, error)


# def get_Hessian_matrix(f, xs, par=()):
#     """任何给定多元函数，算它在 xs 处的 Hessian 矩阵
#     """
#     d = len(xs)  # 矩阵维数
#     return np.array([[multi_partial(f, [i, j], xs, par) for i in range(d)] for j in range(d)])


# def grad_search(f, init_x0, alpha, max_iter, dfmethod="default", *par):
#     """ 梯度下降找极小值
#     """
#     cur_x = init_x0
#     if dfmethod == "default":
#         for _ in range(max_iter):
#             df = np.array([partial_deriv(f, i, cur_x, par) for i in range(len(init_x0))])
#             y0 = f(cur_x, *par)
#             print(y0)
#             cur_alpha = alpha
#             n = la.norm(df)
#             df = df / n
#             while True:
#                 if f(cur_x - cur_alpha * df, *par) > y0:
#                     cur_alpha = cur_alpha / 2
#                 else:
#                     cur_x = cur_x - cur_alpha * df
#                     break
#                 if cur_alpha < 1e-8:
#                     break
#             if abs(f(cur_x, *par) - y0) < 1e-6:
#                 break
#     else:
#         for _ in range(max_iter):
#             df = dfmethod(cur_x, *par)
#             y0 = f(cur_x, *par)
#             print(y0)
#             cur_alpha = alpha
#             n = la.norm(df)
#             df = df / n
#             while True:
#                 if f(cur_x - cur_alpha * df, *par) > y0:
#                     cur_alpha = cur_alpha / 2
#                 else:
#                     cur_x = cur_x - cur_alpha * df
#                     break
#                 if cur_alpha < 1e-8:
#                     break
#             if abs(f(cur_x, *par) - y0) < 1e-6:
#                 break
#     return cur_x


def rd_simple_mat(size, seed=None):
    """随机一个实的单阵，每个矩阵元在 -1 到 1 之间随机。
    - 以概率 1 可相似对角化
    - 以概率 1 不可以酉相似对角化
    - 概率 1 可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 每个矩阵元在 -1 到 1 之间随机的实单阵
    """
    if seed is not None:
        np.random.seed(seed=seed)
    return 2 * np.random.rand(size, size) - 1


def rd_sym_mat(size, seed=None):
    """随机一个实对称矩阵，A.T = A，每个矩阵元在 -1 到 1 之间随机。
    - 必可以酉相似对角化
    - 概率 1 可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 每个矩阵元在 -1 到 1 之间的随机实对称阵
    """
    a = rd_simple_mat(size, seed)
    return a + a.T


def rd_orth_mat(size, seed=None):
    """生成一个实正交阵 A.T @ A = A @ A.T = I。
    - 必可以酉相似对角化
    - 概率 1 可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 实对称矩阵
    """
    a = rd_simple_mat(size, seed)
    q, _ = la.qr(a)
    return q


def rd_singular_mat(size, seed=None):
    """生成一个奇异矩阵，不存在 u, s.t. u^-1 @ A @ u = D。
    - 不能被相似对角化（用来对角化的矩阵的逆矩阵矩阵元发散。）
    - 不能被酉相似对角化
    - 概率 1 可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 奇异矩阵
    """
    if seed is not None:
        np.random.seed(seed=seed)
    n1 = np.array([np.random.randint(2) for i in range(size - 1)])
    while np.all(n1 == 0):
        n1 = np.array([np.random.randint(2) for i in range(size - 1)])
    n = [2 * np.random.random() - 1 for i in range(size)]
    for i in range(size - 1):
        if n1[i] == 1:
            n[i + 1] = n[i]
    a = np.diag(n) + np.diag(n1, 1)
    u = rd_simple_mat(size, seed)
    return la.inv(u) @ a @ u


def rd_simple_mat_complex(size, seed=None):
    """生成一个复的单阵，每个矩阵元的实部和虚部都在 -1 到 1 之间随机。
    - 以概率 1 可相似对角化;
    - 以概率 1 不可酉相似对角化
    - 概率 1 可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 生成一个复的单阵
    """
    if seed is not None:
        np.random.seed(seed=seed)
    return (2 * np.random.rand(size, size) - 1) + 1j * (2 * np.random.rand(size, size) - 1)


def rd_normal_mat_complex(size, seed=None):
    """生成一个正规阵，A.H @ A = A @ A.H
    - 必可以相似对角化
    - 必可以酉相似对角化。
    - 概率 1 可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 正规阵
    """
    if seed is not None:
        np.random.seed(seed=seed)
    v = np.random.random(size) + 1j * np.random.random(size)
    u = rd_unitary_mat(size, seed)
    return u @ np.diag(v) @ u.conj().T


def rd_herm_mat(size, pos_def=False, seed=None):
    """随机一个 Hermite 矩阵，A.H = A
    即本征值为实的正规阵:
        - 必可以相似对角化
        - 必可以酉相似对角化
        - 概率 1 可逆

    Args:
        size (int): 维数
        pos_def (bool, optional): 是否正定（只有 Hermite 矩阵才能讨论正定的问题，一般的单阵本征值为实数并没有特殊的性质，并不定义正定）. Defaults to False.
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: Hermite 矩阵
    """
    a = rd_simple_mat_complex(size, seed)
    a = 0.5 * (a + a.conj().transpose())
    if pos_def:
        np.fill_diagonal(a, np.abs(a.diagonal()) + np.sqrt(2) * size)
    return a


def rd_unitary_mat(size, seed=None):
    """随机幺正矩阵，A.H @ A = A @ A.H = I
    - 必可以相似对角化
    - 必可以酉相似对角化
    - 概率 1 可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 幺正矩阵
    """
    return la.expm((-1.0j * rd_herm_mat(size, seed)))


def rd_noninv_mat(size, seed=None):
    """随机不可逆单阵
    - 必不可逆 或者 逆矩阵发散
    - 以概率 1 可相似对角化
    - 以概率 1 不可酉相似对角化

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 幺正矩阵
    """
    if seed is not None:
        np.random.seed(seed=seed)
    n = np.random.randint(1, size)
    v = np.array([np.random.random() for i in range(size)])
    poslis = list(range(n))
    for _ in range(n):
        v[poslis.pop(np.random.randint(len(poslis)))] = 0
    u = rd_simple_mat(size, seed)
    return la.inv(u) @ np.diag(v) @ u


def rd_realeig_mat(size, seed=None):
    """随机实本征值单阵
    - 以概率 1 可相似对角化
    - 以概率 1 不可酉相似对角化
    - 以概率 1 不可逆

    Args:
        size (int): 维数
        seed (int, optional): 随机数种子。Defaults to None.

    Returns:
        numpy.ndarray: 幺正矩阵
    """
    if seed is not None:
        np.random.seed(seed=seed)
    v = np.array([np.random.random() for _ in range(size)])
    u = rd_simple_mat(size, seed)
    return la.inv(u) @ np.diag(v) @ u

def sym_to_mma(symobj):
    """符号转换为 mma 的工具

    Args:
        symobj (sym): sym 中的符号

    Returns:
        string: mma 中的符号
    """
    # if isinstance(symobj,sym.matrices.dense.MutableDenseMatrix):
    #     symobj = str(symobj)[7:-1]
    new_res = ""
    if isinstance(symobj, str):
        res = symobj
        ct = 0
        ctlist = []
        for ind, i in enumerate(res):
            if i == "[":
                new_res += "{"
            elif i == "]":
                new_res += "}"
            elif i == "s" and ind < len(res) - 6:
                if res[ind:ind + 4] == "sqrt" or res[ind:ind + 3] == "sin":
                    new_res += "S"
                else:
                    new_res += "s"
            elif i == "e" and ind < len(res) - 5:
                if res[ind:ind + 3] == "exp":
                    new_res += "E"
                else:
                    new_res += "e"
            elif i == "c" and ind < len(res) - 5:
                if res[ind:ind + 3] == "cos":
                    new_res += "C"
                else:
                    new_res += "c"
            elif i == "t" and ind < len(res) - 5:
                if res[ind:ind + 3] == "tan":
                    new_res += "T"
                else:
                    new_res += "t"
            elif i == "(":
                if res[ind - 4:ind] == "sqrt" or res[ind:ind + 3] == "sin":
                    new_res += "["
                    ctlist.append(ct)
                elif res[ind - 3:ind] == "exp":
                    new_res += "["
                    ctlist.append(ct)
                elif res[ind - 3:ind] == "cos":
                    new_res += "["
                    ctlist.append(ct)
                elif res[ind - 3:ind] == "tan":
                    new_res += "["
                    ctlist.append(ct)
                else:
                    new_res += "("
                ct += 1
            elif i == ")":
                ct -= 1
                if ct in ctlist:
                    new_res += "]"
                else:
                    new_res += ")"
            elif i == "*" and ind < len(res) - 1:
                if res[ind + 1] == "*":
                    new_res += "^"
                elif res[ind - 1] != "*":
                    new_res += "*"
            else:
                new_res += i
    return new_res


"""for test the above random matrix
N = 3
np.random.seed(seed = None)

# a = pl.rd_simple_mat(N)
# a = pl.rd_sym_mat(N)
# a = pl.rd_orth_mat(N)
# a = pl.rd_sigular_mat(N)    
# a = pl.rd_simple_mat_complex(N)
# a = pl.rd_normal_mat_complex(N)
# a = pl.rd_herm_mat(N)
# a = pl.rd_unitary_mat(N)
# a = pl.rd_noninv_mat(N)
# a = pl.rd_realeig_mat(N)

display(Latex(r"$ A^{-1}: $"))
print(la.inv(a))
print()
v,w = la.eig(a)
print("eig vec:")
print(v)
print()
display(Latex(r"$U^{-1} A U:$"))
print(np.round(la.inv(w) @ a @ w ,2))
display(Latex(r"$U^{\dagger} A U:$"))
print( np.round(w.conj().T @ a @ w ,2))
"""
