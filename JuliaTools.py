from julia import Main
from julia import SparseArrays
from julia import LinearAlgebra
import numpy as np


def get_spectral_density(x_list, coo_matrix):
    mat = SparseArrays.sparse(coo_matrix.row+1, coo_matrix.col+1, coo_matrix.data, mat.shape[1], mat.shape[2])
    res_x = []
    res_y = []
    for x in x_list:
        try:
            num = LinearAlgebra.ldlt(mat,shift=-x)
            res_x.append(res_x,x)
            res_y.append(res_y,np.sum(np.diag(num)<0))
        except:
            continue

if __name__ == "__main__":
    import PyLib.EDtools as qts
    import quspin.basis as qspinbasis
    L = 8
    basis = qspinbasis.spin_basis_1d(L=L,Nup=L//2)
    Delta = 0.1
    a = sum(qts.xx(i,i+1) + qts.yy(i,i+1) + Delta * qts.zz(i,i+1) for i in range(L-1))
    a.get_matrix(basis=basis, dtype=np.float64)
    mat = a.tocsc().tocoo()
    mat = SparseArrays.sparse(mat.row+1, mat.col+1, mat.data, mat.shape[0], mat.shape[1])
    print(type(mat))
    LinearAlgebra.ldlt(mat)
