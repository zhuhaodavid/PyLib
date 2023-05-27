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

from PyLib.EDtools import I, SP, SM, X, Y, Z, N, NN, ZZ, MP, PM, XX, YY, XY, YX
from PyLib.EDtools import I_b, Bdag, B, N_b, PM_b, MP_b, NN_b
from PyLib.EDtools import I_f, Fdag, F, Z_f, N_f, PM_f, MP_f, NN_f, ZZ_f
from PyLib.EDtools import pauli_oper, Oper, show_state, show_basis, show_eig, to_vector, pertub, show_in_another_spin_coord, eig, eigh, eigvals, eigvalsh, eigs, eigsh

np.set_printoptions(linewidth=np.inf, suppress=True)

# import PyLib.BasicTools as bt
# import PyLib.PlotTools as pt
# import PyLib.RandomTools as rt

