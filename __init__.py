"""
EDtools
------------
快速的 ED 工具

LevelStatics
-----------
- 能谱统计

NumTools
-----------
- 产生随机矩阵

Models
----------
自旋模型的求解

PlotTools
----------
画图的函数
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from quspin.basis import spin_basis_1d, boson_basis_1d, spinless_fermion_basis_1d, \
    tensor_basis  # Hilbert space spin basis
from quspin.operators import hamiltonian, quantum_LinearOperator

from PyLib.EDtools import I, sp, sm, x, y, z, n, nn, zz, mp, pm, xx, yy, xy, yx
from PyLib.EDtools import I_b, bdag, b, n_b, pm_b, mp_b, nn_b
from PyLib.EDtools import I_f, fdag, f, z_f, n_f, pm_f, mp_f, nn_f, zz_f
from PyLib.EDtools import eig, eigh, eigvals, eigvalsh, eigs, eigsh
from PyLib.EDtools import show_state, show_basis, show_eig, to_vector, show_in_another_spin_coord
from PyLib.EDtools import ham_heis, ham_ising, ham_XXZ, ham_XY, ham_ZZ_linear
from PyLib.EDtools import BdG_clist, BdG_gd_state, BdG_freefermion, gdenergy_XY_infinite, gdenergy_heisenberg_pbc_approx
from PyLib.EDtools import kron, pauli_oper, pertub

np.set_printoptions(linewidth=np.inf, suppress=True)

# import PyLib.BasicTools as bt
# import PyLib.PlotTools as pt
# import PyLib.RandomTools as rt

