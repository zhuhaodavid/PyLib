import numpy as _np
import scipy.linalg as _la

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
        _np.random.seed(seed=seed)
    return 2 * _np.random.rand(size, size) - 1


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
    q, _ = _la.qr(a)
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
        _np.random.seed(seed=seed)
    n1 = _np.array([_np.random.randint(2) for i in range(size - 1)])
    while _np.all(n1 == 0):
        n1 = _np.array([_np.random.randint(2) for i in range(size - 1)])
    n = [2 * _np.random.random() - 1 for i in range(size)]
    for i in range(size - 1):
        if n1[i] == 1:
            n[i + 1] = n[i]
    a = _np.diag(n) + _np.diag(n1, 1)
    u = rd_simple_mat(size, seed)
    return _la.inv(u) @ a @ u


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
        _np.random.seed(seed=seed)
    return (2 * _np.random.rand(size, size) - 1) + 1j * (2 * _np.random.rand(size, size) - 1)


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
        _np.random.seed(seed=seed)
    v = _np.random.random(size) + 1j * _np.random.random(size)
    u = rd_unitary_mat(size, seed)
    return u @ _np.diag(v) @ u.conj().T


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
        _np.fill_diagonal(a, _np.abs(a.diagonal()) + _np.sqrt(2) * size)
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
    return _la.expm((-1.0j * rd_herm_mat(size, seed)))


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
        _np.random.seed(seed=seed)
    n = _np.random.randint(1, size)
    v = _np.array([_np.random.random() for i in range(size)])
    poslis = list(range(n))
    for _ in range(n):
        v[poslis.pop(_np.random.randint(len(poslis)))] = 0
    u = rd_simple_mat(size, seed)
    return _la.inv(u) @ _np.diag(v) @ u


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
        _np.random.seed(seed=seed)
    v = _np.array([_np.random.random() for _ in range(size)])
    u = rd_simple_mat(size, seed)
    return _la.inv(u) @ _np.diag(v) @ u

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
