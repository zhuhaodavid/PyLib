"""快速的 ED 工具，与其它工具保持独立

快速地生成算符：
- 自旋类算符:
`sp`,`sm`: 自旋上升下降算符.
`x`,`y`,`z`,`n`: 自旋 xyzmn 算符.
`xx`,`yy`,`zz`,`nn`,`mp`,`pm`: 自旋两体算符.
- 波色算符:
`b`,`bdag`,`n_b`: 波色产生湮灭算符、粒子数算符.
`pm_b`, `mp_b`, `nn_b`: bdag b 与 b bdag 算符、两体算符、四体算符.
- 费米算符:
`f`,`fdag`,`n_f`,`z_f`: 费米产生湮灭算符、粒子数算符、z 算符.
`jp_f`, `jp_inv_f`, `nn_f`: fdag f 与 f fdag 算符、两体算符、四体算符.
- `Id`: 单位算符，可以选填 spin, boson, fermion.

生成的算符 (`Oper`类) 的内置功能：
- 快速的生成矩阵：`Oper.get_matrix(basis)` (或者更快速的`Oper.get_matrix(s_par,b_par,f_par)`).
- 本征值分解：`Oper.get_eig([k=1, which='SA'])`或``Oper.get_eigval([k=1, which='SA'])``，不填参数表示计算全谱，分解的结果储存在`Oper.eigenvalues`和`Oper.eigenstates`中.
- 可视化展示本征值、本征态：`Oper.show_eig([n=1,dn=3])`, `Oper.show_eigvalue([dn=3])`，前者需要在给定 quspin 的 basis 下才能工作.
- 显示算符作用之后得到的态：`Oper.action_on(state)`，算符对任何态平均`Oper.expect(state)`

其它的工具
- 从 z 表象转换成其它表象`show_in_another_spin_coord(basis, state, new)`
"""
import numpy as _np
import scipy.linalg as _la
import scipy.sparse.linalg as _sla
import scipy.sparse as _sparse
import itertools
import numbers as _numbers
import copy as _copy


from quspin.operators import hamiltonian, quantum_LinearOperator, exp_op
from quspin.basis import (
    spin_basis_1d,
    boson_basis_1d,
    spinless_fermion_basis_1d,
    tensor_basis,
)


def I(i=0):
    return Oper([["I", [[1, i]]]], ["s"])


def SP(i=0):
    return Oper([["+", [[1, i]]]], ["s"])


def SM(i=0):
    return Oper([["-", [[1, i]]]], ["s"])


def X(i=0):
    return Oper([["x", [[1, i]]]], ["s"])


def Y(i=0):
    return Oper([["y", [[1, i]]]], ["s"])


def Z(i=0):
    return Oper([["z", [[1, i]]]], ["s"])


def N(i=0):
    return Oper([["+-", [[1, i, i]]]], ["s"])


def NN(i, j):
    return Oper([["+-+-", [[1, i, i, j, j]]]], ["s"])


def ZZ(i, j):
    return Oper([["zz", [[1, i, j]]]], ["s"])


def MP(i, j):
    return Oper([["+-", [[1, i, j]]]], ["s"])


def PM(i, j):
    return Oper([["-+", [[1, i, j]]]], ["s"])


def XX(i, j):
    return Oper([["xx", [[1, i, j]]]], ["s"])


def YY(i, j):
    return Oper([["yy", [[1, i, j]]]], ["s"])


def XY(i, j):
    return Oper([["xy", [[1, i, j]]]], ["s"])


def YX(i, j):
    return Oper([["yx", [[1, i, j]]]], ["s"])


def I_b(i=0):
    return Oper([["I", [[1, i]]]], ["b"])


def Bdag(i=0):
    return Oper([["+", [[1, i]]]], ["b"])


def B(i=0):
    return Oper([["-", [[1, i]]]], ["b"])


def N_b(i=0):
    return Oper([["n", [[1, i]]]], ["b"])


def Z_b(i=0):
    return Oper([["z", [[1, i]]]], ["b"])


def PM_b(i, j):
    return Oper([["+-", [[1, i, j]]]], ["b"])


def MP_b(i, j):
    return Oper([["-+", [[1, i, j]]]], ["b"])


def NN_b(i, j):
    return Oper([["nn", [[1, i, j]]]], ["b"])


def ZZ_b(i, j):
    return Oper([["zz", [[1, i, j]]]], ["b"])


def I_f(i=0):
    return Oper([["I", [[1, i]]]], ["f"])


def Fdag(i=0):
    return Oper([["+", [[1, i]]]], ["f"])


def F(i=0):
    return Oper([["-", [[1, i]]]], ["f"], spinless_fermion_basis_1d(L=i + 1))


def X_f(i=0):
    return Oper([["x", [[1, i]]]], ["f"])


def Y_f(i=0):
    return Oper([["y", [[1, i]]]], ["f"])


def Z_f(i=0):
    return Oper([["z", [[1, i]]]], ["f"])


def N_f(i=0):
    return Oper([["n", [[1, i]]]], ["f"])


def PM_f(i, j):
    return Oper([["+-", [[1, i, j]]]], ["f"])


def MP_f(i, j):
    return Oper([["-+", [[1, i, j]]]], ["f"])


def NN_f(i, j):
    return Oper([["nn", [[1, i, j]]]], ["f"])


def XX_f(i, j):
    return Oper([["xx", [[1, i, j]]]], ["f"])


def YY_f(i, j):
    return Oper([["yy", [[1, i, j]]]], ["f"])


def ZZ_f(i, j):
    return Oper([["zz", [[1, i, j]]]], ["f"])


class Oper:
    def __init__(self, static, otype, dynamic=[]):
        self.type = otype  # list
        self.static = static  # numpy.ndarray
        self.dynamic = dynamic  # []
        self.basis = None
        self.Ns = None
        self.isherm = None
        self.pbcwarning = True
        self._pbc_save = None

    def __add__(self, oper2):
        if oper2 == 0:
            return self
        elif isinstance(oper2, _numbers.Number):  # 加数
            appendin = True
            newdata = self.static.copy()
            for i, [oper, coef] in enumerate(newdata):
                if oper == "I":
                    appendin = False
                    coef[0][0] += oper2
                    if _np.isclose(coef[0][0], 0):
                        newdata.pop(i)
                    break
            if appendin:
                newdata.append(["I", [[oper2, 0]]])
            print(newdata)
            return Oper(newdata, self.type)
        elif self.type == oper2.type:  # 相同类型相加 如 ["s"] + ["s"]
            newdata = self.static.copy()
            for term_in_2, coef2 in oper2.static:
                appendin1 = True  # 添加新的算符项
                for term_in_1, coef1 in newdata:
                    if term_in_1 == term_in_2:  # 统一中算符
                        for coef2i in coef2:
                            appendin2 = True  # 添加新的位置
                            for i, coef1i in enumerate(coef1):
                                if coef2i[1:] == coef1i[1:]:
                                    coef1i[0] += coef2i[0]
                                    if _np.isclose(coef1i[0], _np.real(coef1i[0])):
                                        coef1i[0] = _np.real(coef1i[0])
                                    appendin2 = False
                                    if _np.isclose(coef1i[0], 0):
                                        coef1.pop(i)  # 清楚如 ["x", [0, 1]] 这样项
                                    break
                            if appendin2:
                                coef1.append(coef2i)
                        appendin1 = False
                        break
                if appendin1:
                    newdata.append([term_in_2, coef2])
            newdata = [
                datai for datai in newdata if datai[1] != []
            ]  # 清楚如 ["x", []] 这样的项
            new_oper = Oper(newdata, self.type)
            return new_oper
        elif len(self.type) == len(oper2.type) == 1:  # 不同类型的算符相加，但每个算符不是张量基
            typelist = ["b", "s", "f"]
            i1 = typelist.index(self.type[0])
            i2 = typelist.index(oper2.type[0])
            if i1 < i2:
                o1, o2 = self, oper2
            else:
                o2, o1 = self, oper2
            newdata1 = o1.static.copy()
            newdata2 = o2.static.copy()
            for i in newdata1:
                i[0] = i[0] + "|"
            for i in newdata2:
                i[0] = "|" + i[0]
            return Oper(newdata1 + newdata2, o1.type + o2.type)
        elif len(self.type) == 1 or len(oper2.type) == 1:  # 有一个算符非张量基矢
            if len(self.type) == 1:
                o1, o2 = self, oper2
            else:
                o2, o1 = self, oper2  # o1 始终是非张量基
            if o1.type[0] in o2.type:
                newdata = o1.static.copy()
                a = o2.type.index(o1.type[0])  # 找到 o1 在 o2 中的 index
                for newdata1 in newdata:
                    newdata1[0] = "|" * a + newdata1[0] + "|" * (len(o2.type) - a - 1)
                return o2 + Oper(newdata, o2.type)
            else:
                newdata1 = o1.static.copy()
                newdata2 = o2.static.copy()
                for i in newdata1:
                    i[0] = "|" * (len(o2.type)) + i[0]
                for i in newdata2:
                    i[0] = i[0] + "|"
                return Oper(newdata2 + newdata1, o2.type + o1.type)
        else:
            raise NotImplementedError("张量基矢的相加")

    def __radd__(self, oper2):
        if oper2 == 0:
            return self
        return self.__add__(oper2)

    def __sub__(self, oper2):
        return self.__add__((-1) * oper2)

    def __neg__(self):
        return (-1) * self

    def __truediv__(self, other):
        return self.__div__(other)

    def __div__(self, num):
        return (1 / num) * self

    def __rsub__(self, oper2=None):
        return oper2.__add__((-1) * self)

    def __mul__(self, scale):
        if isinstance(scale, Oper):
            return self.__matmul__(scale)
        if scale == 0:
            return 0
        newdata = self.static.copy()
        for i, j in newdata:
            for ji in j:
                ji[0] *= scale
        return Oper(newdata, self.type)

    def __rmul__(self, scale):
        if isinstance(scale, Oper):
            return self.__matmul__(scale)
        return self * scale

    def __matmul__(self, oper2):
        if self.type == oper2.type and len(oper2.type) == 1:  # 相同类型
            newdata = []
            for opername1, coef1 in self.static:
                for opername2, coef2 in oper2.static:
                    newdatai_name = opername1 + opername2
                    newdatai_coef = []
                    for coef1i in coef1:
                        for coef2i in coef2:
                            newdatai_coefi = [coef1i[0] * coef2i[0]]
                            newdatai_coefi += coef1i[1:]
                            newdatai_coefi += coef2i[1:]
                            newdatai_coef.append(newdatai_coefi)
                    newdata.append([newdatai_name, newdatai_coef])
            return Oper(newdata, self.type)
        elif len(self.type) == len(oper2.type) == 1:  # 不同类型
            typelist = ["b", "s", "f"]
            i1 = typelist.index(self.type[0])
            i2 = typelist.index(oper2.type[0])
            if i1 < i2:
                o1, o2 = self, oper2
            else:
                o2, o1 = self, oper2
            newdata = []
            for opername1, coef1 in o1.static:
                for opername2, coef2 in o2.static:
                    newoper = _copy.copy(opername1) + "|"
                    newoper += opername2
                    newcoef = []
                    for coef2i in coef2:
                        for coef1i in coef1:
                            newcoefi = []
                            newcoefi.append(coef1i[0] * coef2i[0])
                            newcoefi += coef1i[1:]
                            newcoefi += coef2i[1:]
                            newcoef.append(newcoefi)
                    newdata.append([newoper, newcoef])
            return Oper(newdata, [o1.type[0], o2.type[0]])

    def __pow__(self, n, m=None):
        """
        POWER operation.
        """
        if m is not None:
            raise NotImplementedError("modulo is not implemented")
        newoper = self
        if n <= 0:
            raise NotImplementedError("inverse is not implemented")
        for _ in range(n - 1):
            newoper = self * newoper
        return newoper

    def expand(self, pauli=0):
        if self.basis is None:
            tmpbasis = []
            for otype in self.type:
                if otype == "b":
                    tmpbasis.append(boson_basis_1d(L=1, sps=4))
                elif otype == "s":
                    tmpbasis.append(spin_basis_1d(L=1, pauli=pauli))
                elif otype == "f":
                    tmpbasis.append(spinless_fermion_basis_1d(L=1))
                else:
                    raise NotImplementedError()
            if len(tmpbasis) == 1:
                tmpbasis = tmpbasis[0]
            else:
                tmpbasis = tensor_basis(*tmpbasis)
        else:
            tmpbasis = self.basis
        newoper = Oper(
            tmpbasis.expanded_form(self.static, self.dynamic)[0], otype=self.type
        )
        newoper.__basis = self.basis
        return newoper

    def check_hermitian(self):
        if self.basis is None:
            tmpbasis = []
            for otype in self.type:
                if otype == "b":
                    tmpbasis.append(boson_basis_1d(L=1, sps=4))
                elif otype == "s":
                    tmpbasis.append(spin_basis_1d(L=1, pauli=0))
                elif otype == "f":
                    tmpbasis.append(spinless_fermion_basis_1d(L=1))
                else:
                    raise NotImplementedError()
            if len(tmpbasis) == 1:
                tmpbasis = tmpbasis[0]
            else:
                tmpbasis = tensor_basis(*tmpbasis)
        else:
            tmpbasis = self.basis
        static_list, dynamic_list = tmpbasis._get_local_lists(
            static=self.static, dynamic=self.dynamic
        )
        static_expand, static_expand_hc, _, _ = tmpbasis._get_hc_local_lists(
            static_list, dynamic_list
        )
        diff = set(tuple(static_expand)) - set(tuple(static_expand_hc))
        if diff:
            return False
        else:
            return True

    def check_basis(self, basis=None, s_par=None, b_par=None, f_par=None, pauli=0):
        if basis is not None:
            if type(basis) in [
                spin_basis_1d,
                boson_basis_1d,
                tensor_basis,
                spinless_fermion_basis_1d,
            ]:
                self.basis = basis
            else:
                raise Exception("这个基矢不支持")
            if pauli is not None:
                basislist = get_tensor_basis_list(basis)
                for basisi in basislist:
                    if type(basisi) == spin_basis_1d and pauli != basis._pauli:
                        print("输入的 pauli 与 basis._pauli 不同，忽略 pauli ")
        elif s_par is not None or b_par is not None or f_par is not None:
            basis = []
            if pauli is None:
                pauli = 0
            for i in self.type:
                if i == "b" and b_par is not None:
                    basis.append(boson_basis_1d(L=b_par[0], sps=b_par[1]))
                elif i == "s" and s_par is not None:
                    if isinstance(s_par, int):  # 如果是整数作为链长输入
                        basis.append(spin_basis_1d(L=s_par, S="1/2", pauli=pauli))
                    else:  # 否则可以指定自旋
                        j = (
                            str(s_par[1])
                            if isinstance(s_par[1], int)
                            else str(int(s_par[1] * 2)) + "/2"
                        )
                        basis.append(spin_basis_1d(L=s_par[0], S=j, pauli=pauli))
                elif i == "f" and f_par is not None:
                    basis.append(spinless_fermion_basis_1d(L=f_par[0]))
                else:
                    raise Exception("算符与参数不匹配")
            if self.basis is not None:
                print("覆盖原有的 basis")
            if len(basis) == 1:
                self.basis = basis[0]
            else:
                self.basis = tensor_basis(*basis)
        elif self.basis is None:
            raise Exception("需要 basis 或 par ")
        elif type(self.basis) not in [
            spin_basis_1d,
            boson_basis_1d,
            tensor_basis,
            spinless_fermion_basis_1d,
        ]:
            raise Exception("这个基矢不支持")
        self.isherm = self.check_hermitian()
        self.Ns = self.basis.Ns

    def get_matrix(
        self,
        basis=None,
        s_par=None,
        b_par=None,
        f_par=None,
        dtype=_np.complex128,
        pauli=None,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    ):
        """生成算符的矩阵表示
        方法 1 的优先级高于 方法 2

        Args:
        ---------
            - 方法 1：

            `basis`:  quspin 的 basis，支持`spin_basis_1d`,`boson_basis_1d`,`spinless_fermion_basis_1d`,`tensor_basis`.
            `dtype`, `check_symm`, `check_herm`,`check_pcon`: 同 quspin 的参数.

            - 方法 2：

            `s_par` (int): 自旋的格点数目.
            `b_par` (list): [波色的格点数，每个格点上的截断数 int/list].
            `f_par` (int): 费米的格点数目.
            `am_par` (list): [格点数目，每个格点上的角量子数 int/list].
            `pauli` (0/-1): 是否采用泡利算符，0 表示不采用/用自旋算符，-1 表示采用，
        """
        self.check_basis(basis, s_par, b_par, f_par, pauli=pauli)
        return hamiltonian(
            self.static,
            self.dynamic,
            basis=self.basis,
            dtype=dtype,
            check_symm=check_symm,
            check_herm=check_herm,
            check_pcon=check_pcon,
        )

    def get_LinearOperator(
        self,
        basis=None,
        s_par=None,
        b_par=None,
        f_par=None,
        dtype=_np.complex128,
        pauli=0,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    ):
        """生成线性算符，会减慢速度，但能极大节约内存

        Args:
        ---------
            - 方法 1：

            `basis`:  quspin 的 basis，支持`spin_basis_1d`,`boson_basis_1d`,`spinless_fermion_basis_1d`,`tensor_basis`.
            `dtype`, `check_symm`, `check_herm`,`check_pcon`: 同 quspin 的参数.

            - 方法 2：

            `s_par` (int): 自旋的格点数目.
            `b_par` (list): [波色的格点数，每个格点上的截断数 int/list].
            `f_par` (int): 费米的格点数目.
            `am_par` (list): [格点数目，每个格点上的角量子数 int/list].
            `pauli` (0/-1): 是否采用泡利算符，0 表示不采用/用自旋算符，-1 表示采用，none 表示
        """
        self.check_basis(basis, s_par, b_par, f_par, pauli)
        return quantum_LinearOperator(
            self.static,
            basis=self.basis,
            dtype=dtype,
            check_symm=check_symm,
            check_herm=check_herm,
            check_pcon=check_pcon,
        )

    def show_oper(self, fmt="default"):
        """显示算符，有 fancy = False/True 两种选择，pauli 不建议指定，不指定时将根据 self.basis 自动选择"""
        if fmt == "itensor":
            for oper, coef in self.static:
                oper_tmp = [i for i in oper if i != "|"]
                for coefi, *posi in coef:
                    print(
                        "  "
                        + str(coefi)
                        + ", "
                        + ", ".join([i + str(j) for (i, j) in zip(oper_tmp, posi)])
                    )
        else:
            for oper, coef in self.static:
                print(oper)
                for coefi, *posi in coef:
                    print("  ", posi, coefi)

    def show(self, eular_form=False, d=6, pauli=None):
        """显示算符，有 fancy = False/True 两种选择，pauli 不建议指定，不指定时将根据 self.basis 自动选择

        Args:
            fancy (bool, optional): 是否显示 Latex 格式。Defaults to False.
            eular_form (bool, optional): 复数的显式格式。Defaults to True.
            d (int, optional): 保留小数维数。Defaults to 6.
            pauli (_type_, optional): 是否使用 pauli 算符。Defaults to None.
        """
        from IPython.display import display, Latex

        res = ""
        if pauli is not None:
            pass
        elif self.basis is None:
            pauli = 0
        else:
            if type(self.basis) == tensor_basis:
                basis_list = get_tensor_basis_list(self.basis)
            else:
                basis_list = [self.basis]
            for basis in basis_list:
                if type(basis) == spin_basis_1d:
                    pauli = basis._pauli
        for opers, coef in self.static:  # 不同的项
            for coefi in coef:
                temp = show_coeff_fuc(coefi[0], d, eular_form)
                tag1 = False
                if temp == "+ " or temp == "+ ":
                    tag1 = True
                res += temp
                tag2 = True
                temp = self.show_oper_single_latex(opers, coefi[1:], self.type, pauli)
                if temp != "":
                    tag2 = False
                res += temp
                if tag1 and tag2:
                    res += "1"
        if res[0] == "+":
            res = res[1:]
        display(Latex("$" + res + "$"))

    def show_oper_single_latex(self, opers, pos, oper_type, pauli):
        res = ""
        spin_oper = "s" if pauli == 0 else "\sigma"
        oper_type_ind = 0
        oper_ind = 0
        for oper in opers:
            if oper == "|":
                oper_type_ind += 1
                continue
            if oper_type[oper_type_ind] == "s":
                if oper == "I":  # 1
                    res += ""
                elif oper == "+":  # 1
                    res += spin_oper + "^+_{" + str(pos[oper_ind]) + "}"
                elif oper == "-":  # 2
                    res += spin_oper + "^-_{" + str(pos[oper_ind]) + "}"
                elif oper == "x":  # 6
                    res += spin_oper + "^x_{" + str(pos[oper_ind]) + "}"
                elif oper == "y":  # 7
                    res += spin_oper + "^y_{" + str(pos[oper_ind]) + "}"
                elif oper == "z":  # 8
                    res += spin_oper + "^z_{" + str(pos[oper_ind]) + "}"
                elif oper == "n":  # 9
                    res += (
                        "\\frac{" + spin_oper + "^z_{" + str(pos[oper_ind]) + "}+1}{2}"
                    )
            elif oper_type[oper_type_ind] == "b":
                if oper == "I":  # 1
                    res += ""
                elif oper == "+":  # 1
                    res += "b^{\dagger}_{" + str(pos[oper_ind]) + "}"
                elif oper == "-":  # 2
                    res += "b_{" + str(pos[oper_ind]) + "}"
                elif oper == "n":  # 5
                    res += (
                        "b^{\dagger}_{"
                        + str(pos[oper_ind])
                        + "}b_{"
                        + str(pos[oper_ind])
                        + "}"
                    )
            elif oper_type[oper_type_ind] == "f":
                if oper == "I":  # 1
                    res += ""
                elif oper == "+":  # 1
                    res += "f^{\dagger}_{" + str(pos[oper_ind]) + "}"
                elif oper == "-":  # 2
                    res += "f_{" + str(pos[oper_ind]) + "}"
                elif oper == "z":  # 3
                    res += "(2n_{" + str(pos[oper_ind]) + "}-1)"
                elif oper == "n":  # 4
                    res += (
                        "f^{\dagger}_{"
                        + str(pos[oper_ind])
                        + "}f_{"
                        + str(pos[oper_ind])
                        + "}"
                    )
            else:
                print(oper_type)
                raise NotImplementedError("算符类型包含" + oper_type[oper_type_ind] + "，无法识别")
            oper_ind += 1
        return res


#######################################
# 生成算符
######################################


def rpmethod(match):
    """快速生成哈密顿量，如 xx + yy 就是 x1 x2 + y1 y2

    Args:
        match (字符串): _description_

    Returns:
        _type_: _description_
    """
    dic = {
        "x": "_np.array([[0.,1.],[1.,0.]]),",
        "y": "_np.array([[0.,-1.j],[1.j,0.]]),",
        "z": "_np.array([[1.,0.],[0.,-1.]]),",
        "P": "_np.array([[0.,1.],[0.,0.]]),",
        "M": "_np.array([[0.,0.],[1.,0.]]),",
        "I": "_np.array([[1.,0.],[0.,1.]]),",
        "x": "_np.array([[0.,0.5],[0.5,0.]]),",
        "y": "_np.array([[0.,-0.5j],[0.5j,0.]]),",
        "z": "_np.array([[0.5,0.],[0.,-0.5]]),",
        "p": "_np.array([[0.,1.],[0.,0.]]),",
        "m": "_np.array([[0.,0.],[1.,0.]]),",
        "i": "_np.array([[1.,0.],[0.,1.]]),",
    }
    res = "_qu.kron("
    for xi in match.group():
        res += dic[xi]
    return res[:-1] + ")"


def pauli_oper(stri):
    import re

    new = re.sub("[xyzpmiXYZPMI]+", rpmethod, stri)
    return eval(new)


#######################################
# 算符操作
######################################
def tocoolist(sparmatrix):
    """julia 中这样调用这个函数
    sparse(H.tojulia()...)

    matlab 中这样调用这个函数
    sparse(res(1,:),res(2,:),res(3,:),double(dim),double(dim))
    """
    tmp = _sparse.coo_matrix(sparmatrix)
    res = _np.zeros((3, len(tmp.row)), dtype=tmp.dtype)
    res[0, :] = tmp.row + 1
    res[1, :] = tmp.col + 1
    res[2, :] = tmp.data
    return res, tmp.shape[0]


def convert2qu(H):
    import quimb as _qu

    @_qu.gen.operators.hamiltonian_builder
    def cvt2csr(x):
        try:
            return x.tocsr()
        except:
            return _sparse.csr_matrix(x)

    return cvt2csr(H)


def eig(H, backend="scipy", **kwarg):
    """一般矩阵本征值分解"""
    if backend == "quimb":
        import quimb as _qu

        quH = H if type(H) == _qu.core.qarray else convert2qu(H)
        return _qu.eig(quH, **kwarg)
    elif backend == "scipy":
        return _la.eig(H.toarray(), **kwarg)
    elif backend == "numpy":
        return _np.linalg.eig(H.toarray(), **kwarg)
    else:
        raise NotImplemented(backend)


def eigh(H, backend="scipy", **kwarg):
    """厄密矩阵本征值分解"""
    if backend == "quimb":
        import quimb as _qu

        quH = H if type(H) == _qu.core.qarray else convert2qu(H)
        return _qu.eigh(quH, **kwarg)
    elif backend == "scipy":
        return _la.eigh(H.toarray(), **kwarg)
    elif backend == "numpy":
        return _np.linalg.eigh(H.toarray(), **kwarg)
    else:
        raise NotImplemented(backend)


def eigvals(H, backend="scipy", **kwarg):
    """一般矩阵本征值分解，但只求本征值"""
    if backend == "quimb":
        import quimb as _qu

        quH = H if type(H) == _qu.core.qarray else convert2qu(H)
        return _qu.eigvals(quH, **kwarg)
    elif backend == "scipy":
        return _la.eigvals(H.toarray(), **kwarg)
    elif backend == "numpy":
        return _np.linalg.eigvals(H.toarray(), **kwarg)
    else:
        raise NotImplemented(backend)


def eigvalsh(H, backend="scipy", **kwarg):
    """厄密矩阵本征值分解，但只求本征值"""
    if backend == "quimb":
        import quimb as _qu

        quH = H if type(H) == _qu.core.qarray else convert2qu(H)
        return _qu.eigvalsh(quH, **kwarg)
    elif backend == "scipy":
        return _la.eigvalsh(H.toarray(), **kwarg)
    elif backend == "numpy":
        return _np.linalg.eigvalsh(H.toarray(), **kwarg)
    else:
        raise NotImplemented(backend)


def svd(H, backend="scipy", **kwarg):
    """一般矩阵奇异值分解"""
    if backend == "quimb":
        import quimb as _qu

        quH = H if type(H) == _qu.core.qarray else convert2qu(H)
        return _qu.svd(quH, **kwarg)
    elif backend == "scipy":
        return _la.svd(H.toarray(), **kwarg)
    elif backend == "numpy":
        return _np.linalg.svd(H.toarray(), **kwarg)
    else:
        raise NotImplemented(backend)


def eigs(H, k, backend="scipy", **kwarg):
    """一般矩阵 lanczos"""
    if backend == "quimb":
        import quimb as _qu

        quH = H if type(H) == _qu.core.qarray else convert2qu(H)
        return _qu.eig(quH, k, **kwarg)
    elif backend == "scipy":
        assert _sparse.issparse(H)
        return _sla.eigs(H, k, **kwarg)
    else:
        raise NotImplemented(backend)


def eigsh(H, k, backend="scipy", **kwarg):
    """厄密矩阵 lanczos"""
    if backend == "quimb":
        import quimb as _qu

        quH = H if type(H) == _qu.core.qarray else convert2qu(H)
        return _qu.eigh(quH, k, **kwarg)
    elif backend == "scipy":
        assert _sparse.issparse(H)
        return _sla.eigsh(H, k, **kwarg)
    else:
        raise NotImplemented(backend)


def exp_action_on(
    operator,
    v,
    a=1.0,
    dtype=None,
    work_array=None,
    overwrite_v=False,
    method="speed_first",
    tol=1e-5,
    m=None,
):
    """计算 expm(a*A) * v 三种方法
    - 'exact' 利用 scipy._sparse.linalg.expm_multiply，需要 matrix
    - 'speed_first' 利用 quspin.expm_multiply_parallel，需要 matrix
    - 'memory_first' 利用 quspin.expm_lanczos，可以用 LinearOperator，但稀疏矩阵千万维的矩阵内存问题不大，但速度问题相比就很大，因此这个 _lanczos 意义并不大

    Args:
        v (numpy.ndarray): 1d or 2d array
        a (float, optional): 乘以 A 的数。Defaults to 1.0.
        dtype (numpy.dtype, optional): a*A 的类型。Defaults to None.
        work_array (numpy.ndarray, optional): 任意 shape 但至少要包含 2 * v.size 的空间，这个预设可以帮助分配内存。Defaults to None.
        overwrite_v (bool, optoinal): 是否改变 v . Defaults to False.
        tol (numpy.float64, optional): 误差。Defaults to None.
        method (str, optional): "speed_first/memory_first/exact". Defaults to "speed_first".
        m (int, optional): _lanczos 维数。Defaults to 20.

    Raises:
        Exception: _description_
        Exception: _description_

    Returns:
        _type_: _description_
    """
    from quspin.tools.evolution import expm_multiply_parallel

    if method == "speed_first" and type(operator) == hamiltonian:
        expA = expm_multiply_parallel(operator.tocsr(), a=a, dtype=dtype)
        return expA.dot(v, work_array=work_array, overwrite_v=overwrite_v)
    elif method == "memory_first" and type(operator) == quantum_LinearOperator:
        from quspin.tools.lanczos import expm_lanczos, _lanczos_iter

        assert abs(_la.norm(v) - 1) < 1e-6
        if m is None:
            v_lanczos_iter = v
            for m in range(20, 50, 5):
                # eps: Used to cutoff _lanczos iteration when off diagonal matrix elements of T drops below this value.
                E, V, Q_T = _lanczos_iter(
                    operator,
                    v,
                    m=m,
                    copy_v0=True,
                    copy_A=False,
                    eps=None,
                )
                new_v_lanczos_iter = expm_lanczos(E, V, Q_T, a=a)
                err = _la.norm(new_v_lanczos_iter - v_lanczos_iter)
                if err < tol:
                    return new_v_lanczos_iter
                v_lanczos_iter = new_v_lanczos_iter
            print("maximum iter_time reached, err =", err)
        else:
            E, V, Q_T = _lanczos_iter(
                operator, v, m=m, copy_v0=True, copy_A=False, eps=None
            )
            v_lanczos_iter = expm_lanczos(E, V, Q_T, a=a)
        return v_lanczos_iter
    elif method == "exact":
        if operator is None:
            raise Exception("没有 get_matrix")
        return _sla.expm_multiply(operator.tocsr(), v)
    else:
        raise NotImplemented(method)


#################################################
#  生成态
#################################################


def find_index(basis, strrepr):
    """给定基矢，字符串表示/或者是 tuple 表示，找到在该基矢组中的 index

    Args:
        basis (quspin.basis): quspin.basis 格式的基矢
        strrepr (str): 字符串或者 tuple 表示

    Returns:
        int: index
    """
    if type(basis) == tensor_basis:
        basis_list = []
        basis_cur = basis
        while type(basis_cur) == tensor_basis:
            basis_list.append(basis_cur.basis_left)
            basis_cur = basis_cur.basis_right
        basis_list.append(basis_cur)
        assert len(basis_list) == len(strrepr)
        Ns_list = [i.Ns for i in basis_list]

        Ss_list = [
            find_index(basis_list[i], strrepr[i]) for i in range(len(basis_list))
        ]
        index = Ss_list[-1]
        for i in range(2, len(Ss_list) + 1):
            index += Ss_list[-i] * _np.prod(Ns_list[-i + 1 :])
        return int(index)
    else:
        if type(strrepr) == str:
            return int(basis.index(strrepr))
        if type(strrepr) == tuple:
            res = 0
            if type(basis) == spin_basis_1d:
                n = 2
            if type(basis) == boson_basis_1d:
                n = basis.sps
            for i in range(1, basis.L + 1):
                res += strrepr[-i] * n ** (i - 1)
            return int(basis.index(res))


def to_vector(basis, strlist, coeflist, dense=True):
    """给定基矢描述，转化成列向量形式

    Args:
        basis (quspin.basis): quspin 的基矢
        strlist (list): 列表，字符串，如 ["1011","1101"] 或者 [(12,13),(9,2)]
        coeflist (list): 每个态的叠加系数如 [1.2, 3.4]
        dense (bool, optional): 是否存为稠密格式。Defaults to True.

    Returns:
        _np.ndarray/scipy._sparse.csr_matrix: 列向量
    """
    if dense:
        vec = _np.zeros(basis.Ns)
        for ind, i in enumerate(strlist):
            vec[find_index(basis, i)] = coeflist[ind]
        return vec
    else:
        vec = _sparse.lil_array((basis.Ns, 1))
        for ind, i in enumerate(strlist):
            vec[find_index(basis, i)] = coeflist[ind]
        return vec.tocsr()


#################################################
#  态的可视化   show_state,  show_basis, show_eig
#################################################
def invert01(x):
    if x == "0":
        return "1"
    elif x == "1":
        return "0"
    else:
        return x


def show_basis_basic(basis, ind=None, tp="plain", prt=True, vrtlist=False):
    """可视化给定 quspin 的非张量基，如果 ind 给定，则只可视化该指标对应的态"""
    up_dic = {"z": "↑", "x": "+", "y": "x"}
    dn_dic = {"z": "↓", "x": "-", "y": "·"}
    if ind is not None:
        intrepr = basis.states[ind]
        strirepr = basis.int_to_state(intrepr, bracket_notation=False)
        if vrtlist and basis.sps == 2:
            strirepr = "".join([invert01(i) for i in strirepr])
        if tp in ["x", "y", "z"] and isinstance(basis, spin_basis_1d):
            res = "|"
            for i in strirepr:
                if i == "1":
                    res += up_dic[tp]
                elif i == "0":
                    res += dn_dic[tp]
                else:
                    res += i
            res += "⟩ "
            return res
        else:
            return "|" + strirepr + "⟩"
    else:
        for i in range(basis.Ns):
            if prt:
                print(show_basis_basic(basis, ind=i, tp=tp, vrtlist=vrtlist))
            else:
                return show_basis_basic(basis, ind=i, tp=tp, vrtlist=vrtlist)


def show_tensor_basis(basis_list, indlist, tp="plain", prt=True, vrtlist=False):
    """可视化 quspin 张量基矢，这个是相对比较核心的函数"""
    ostring = ""
    for ind in range(len(indlist)):
        ostring += show_basis_basic(
            basis_list[ind], indlist[ind], tp=tp, prt=False, vrtlist=vrtlist
        )
    if prt:
        print(ostring)
    else:
        return ostring


def get_tensor_basis_list(basis):
    """给定张量基，分解出它的 list"""
    basis_list = []
    basis_cur = basis
    while type(basis_cur) == tensor_basis:
        basis_list.append(basis_cur.basis_left)
        basis_cur = basis_cur.basis_right
    basis_list.append(basis_cur)
    return basis_list


def show_basis(basis, k=-1, tp="plain", prt=True, vrtlist=False):
    """给定 quspin 的任意基矢，可视化第 k 个，k=-1 表示全部基矢，
    vrtlist (list, optional): 编号是否倒置。Defaults to None.
    """
    if k == -1:
        basis_list = get_tensor_basis_list(basis)
        basis_list_intrepre = tuple(list(range(i.Ns)) for i in basis_list)
        ct = 0
        res = []
        for i in itertools.product(*basis_list_intrepre):
            ct += 1
            res += [show_tensor_basis(basis_list, i, tp=tp, prt=prt, vrtlist=vrtlist)]
        if not prt:
            return res
    else:
        basis_list = get_tensor_basis_list(basis)
        basis_list_intrepre = [i.Ns for i in basis_list]
        statei_list = _split_each_basis_index(basis_list_intrepre, k)
        return show_tensor_basis(
            basis_list, statei_list, tp=tp, prt=prt, vrtlist=vrtlist
        )


def _split_each_basis_index(basis_list_intrepre, i):
    res = []
    tmp = _np.cumprod(basis_list_intrepre[::-1])[:-1]
    for j in tmp[::-1]:
        res.append(i // j)
        i %= j
    res.append(i)
    return res


def show_state(
    basis,
    state,
    threshold=1e-10,
    tp="plain",
    vrtlist=False,
):
    """给定列向量，在 quspin 基矢下可视化

    Args:
        basis (quspin.basis): quspin 的基矢
        state (numpy.ndarray/scipy._sparse.csr_matrix): 向量形式的态
        threshold (float64, optional): 保留大于该数的态。Defaults to 1e-10.
        tp (str, optional): 输出格式 plian/z/x/y. Defaults to "plain".
        prt (bool, optional): 是否输出。Defaults to True.
        vrtlist (list, optional): 编号是否倒置。Defaults to None.

    Returns:
        str: 可视化的态
    """
    basis_list = get_tensor_basis_list(basis)

    if _sparse.issparse(state):
        # 稀疏矩阵形式的态
        state = _sparse.coo_array(state).reshape(-1, 1)
        assert state.shape[0] == basis.Ns
        basis_list_intrepre = [i.Ns for i in basis_list]
        for statei, coefi in zip(state.row, state.data):
            res = ""
            statei_list = _split_each_basis_index(basis_list_intrepre, statei)
            basisstr = show_tensor_basis(
                basis_list, statei_list, tp=tp, prt=False, vrtlist=vrtlist
            )
            res = basisstr + ": " + str(coefi)
            print(res)
    else:
        state = _np.array(state).reshape(-1)
        ct = 0
        basis_list_intrepre = tuple(list(range(i.Ns)) for i in basis_list)
        for i in itertools.product(*basis_list_intrepre):
            res = ""
            if abs(state[ct]) <= threshold:
                ct += 1
                continue
            basisstr = show_tensor_basis(
                basis_list, i, tp=tp, prt=False, vrtlist=vrtlist
            )
            res = basisstr + ": " + str(state[ct])
            print(res)
            ct += 1


def complex_eular_form_np(a, d, eular_form):
    """复数 Eular 表示转换为实部虚部表示

    Args:
        a (complex): 复数
        d (int): 保留位数
        eular_form (bool): 是否转化为 Eular 形式

    Returns:
        str: 复数可视化
    """
    if eular_form:
        ang = _np.angle(a) / _np.pi
        if ang < 0:
            return (
                str(_np.round(abs(a), d)) + "exp{-i" + str(_np.round(-ang, d)) + "\pi}"
            )
        elif ang == 0:
            return str(_np.round(abs(a), d))
        else:
            return str(_np.round(abs(a), d)) + "exp{i" + str(_np.round(ang, d)) + "\pi}"
    else:
        return str(_np.round(a, d))


def show_coeff_fuc(coef, d, eular_form):
    """系数可视化

    Args:
        coef (float/complex): 系数
        d (int): 保留位数
        eular_form (bool): 是否为复数

    Returns:
        str: 可视化的系数
    """
    res = ""
    if _np.imag(coef) != 0:
        res += "+ " + complex_eular_form_np(coef, d, eular_form)
    elif _np.real(coef) < 0:
        coef = _np.real(coef)
        if coef == -1:
            res += "- "
        else:
            coef = -_np.round(coef, d)
            res += "- " + str(coef)
    else:
        coef = _np.real(coef)
        if coef == 1:
            res += "+ "
        else:
            coef = _np.round(coef, d)
            res += "+ " + str(coef)
    return res


def tranf_xyz_mat(basis, new, old=None):
    uzx = _np.array([[1, 1], [1, -1]]) / _np.sqrt(2)
    uxz = _np.array([[1, 1], [1, -1]]) / _np.sqrt(2)
    uzy = _np.array([[1, 1j], [1j, 1]]) / _np.sqrt(2)
    uyz = _np.array([[1, -1j], [-1j, 1]]) / _np.sqrt(2)
    idt = _np.array([[1, 0], [0, 1]])
    uyx = uyz * uzx
    uxy = uxz * uzy
    if isinstance(basis, spin_basis_1d):
        if len(new) == 1:
            new = new * basis.L
        else:
            if len(new) != basis.L:
                raise Exception("Wrong site number")
        if old is None:
            old = "z" * basis.L
        U = []
        for ind, i in enumerate(new):
            if i == old[ind]:
                U.append(idt)
            elif i == "x" and old[ind] == "z":
                U.append(uxz)
            elif i == "z" and old[ind] == "x":
                U.append(uzx)
            elif i == "y" and old[ind] == "z":
                U.append(uyz)
            elif i == "z" and old[ind] == "y":
                U.append(uzy)
            elif i == "x" and old[ind] == "y":
                U.append(uxy)
            elif i == "y" and old[ind] == "x":
                U.append(uyx)
        return _np.kron(*U)
    else:
        raise NotImplementedError("只能用来处理自旋基矢")


def show_eig(
    basis,
    eigenvalues,
    eigenstates,
    n=None,
    valudn=3,
    threshold=1e-10,
    tp="plain",
    vrtlist=False,
):
    """以好看的形式显示本征值和本征向量

    Args:
        n (int, optional): which eigenstate. Defaults to None.
        valudn (int, optional): 本征值保留到几位小数。Defaults to 3.
        threshold (float): 系数小于这个值的态不显示。Defaults to 1e-10
        tp (string, optional): type 显示的类型，可选选择 z, x, y plain。Defaults to "plain"
        vrtlist (list, optional): 编号是否倒置。Defaults to None.
    """
    eigstr = "==== %." + str(valudn) + "f ===="
    if n is None:
        for i in range(len(eigenvalues)):
            if eigenvalues[i] is not None:
                print(eigstr % eigenvalues[i])
                show_state(
                    basis,
                    eigenstates[:, i : i + 1],
                    threshold=threshold,
                    tp=tp,
                    vrtlist=vrtlist,
                )
    else:
        print(eigstr % eigenvalues[n])
        show_state(
            basis,
            eigenstates[:, n : n + 1],
            threshold=threshold,
            tp=tp,
            vrtlist=vrtlist,
        )


def show_in_another_spin_coord(basis, state, new, old=None, d=8, eular_form=True):
    U = tranf_xyz_mat(basis, new, old)
    if len(state.shape) == 1:
        state = _np.array(state, ndmin=2).transpose()
    res = ""
    for ind, i in enumerate(U @ state):
        if i == 0:
            continue
        res += show_coeff_fuc(i[0], d, eular_form) + show_basis(
            basis, ind, tp=new, prt=False
        )
    if res[0] == "+":
        res = res[1:]
    print(res)


#################################################
#  微扰计算
#################################################


def update_u0(lis, u0, mat):
    """简并微扰中重新调整 u0，（注计算过程中会改变 u0 的值）

    Args:
        lis (list): [E0, E1, ...] 各阶微扰的能量本征值
        u0 (numpy.ndarray): 零阶近似下的本征矩阵
        mat (numpy.ndarray): 微扰项 V 在零阶本征态表象下的矩阵

    Returns:
        E_new (numpy.ndarray): 下一阶近似下的能量本征值
        u0 (numpy.ndarray): 更新后的零阶本征矩阵
    """
    last_data_lis = [i[0] for i in lis]
    eig_len = len(lis[0])
    E_new = _np.zeros(eig_len, dtype=mat.dtype)
    same_val_ind = [0]
    for i in range(eig_len):
        tag = True  # 是否直接继续向后查询
        if i == eig_len - 1:
            tag = False
        else:
            data_lis = [j[i + 1] for j in lis]
            for last_data, data in zip(data_lis, last_data_lis):
                if ~_np.isclose(last_data, data):
                    tag = False
        if tag:
            same_val_ind.append(i + 1)
            continue
        elif len(same_val_ind) == 1:
            E_new[same_val_ind] = mat[same_val_ind, same_val_ind]
        else:
            mat_subset = mat[_np.ix_(same_val_ind, same_val_ind)]
            E_new[same_val_ind], mat_subset_vec = _la.eigh(mat_subset)
            u0[:, same_val_ind] = u0[:, same_val_ind] @ mat_subset_vec
        same_val_ind = [i + 1]
        last_data_lis = data_lis
    return E_new, u0


def pertub(E0, U0, V, return_state=False, order=1):
    """计算一阶和二阶微扰

    Args:
        E0 (numpy.ndarray, 1D): 零阶能量本征值
        U0 (numpy.ndarray, 2D): 一阶本征矩阵
        V (quspin.operator, 需要有 V.matrix_ele 这个成员函数): 微扰项
        return_state (bool, optional): 是否返回各阶本征矩阵。Defaults to False.
        order (int, optional): 微扰的阶数。Defaults to 1.

    Raises:
        Exception: 只能计算一阶和二阶微扰，高阶微扰需要考虑费曼图方法

    Returns:
        _type_: _description_
    """
    # 一阶微扰
    v_tilde = V.matrix_ele(U0, U0)  # V 在 H0 下的矩阵元
    E1, U0 = update_u0([E0], U0, v_tilde)  # 没有简并的话可以不用更新，直接写 E1 = np.diag(v_tilde)
    if not return_state and order == 1:
        return E0, E1
    else:
        v_tilde = V.matrix_ele(U0, U0)  # V 在 H0 下的矩阵元
        epsilon = _np.subtract.outer(E0, E0)
        block_diag = _np.isclose(epsilon, 0)
        omega = -(1 / (epsilon + block_diag)) * ~block_diag
        U1 = omega * v_tilde
    if return_state and order == 1:
        return E0, U0, E1, U1
    # 二阶微扰
    else:
        h1u1 = v_tilde.conj() @ U1
        E2, U0 = update_u0([E0, E1], U0, h1u1)
        # 没有简并的话可以不用更新，直接写 E2 = np.sum(v_tilde.conj() * u_tilde, axis=0)
    if not return_state and order == 2:
        return E0, E1, E2
    elif return_state and order == 2:
        v_tilde = V.matrix_ele(U0, U0)
        U1 = omega * v_tilde
        epsilon1 = _np.subtract.outer(E1, E1)
        block_diag1 = _np.isclose(epsilon1, 0)
        U1 += (
            (v_tilde @ U1)
            * _np.isclose(epsilon, 0)
            * (1 / (epsilon1 + block_diag1))
            * ~block_diag1
        )
        U2 = omega * (v_tilde @ U1 - U1 * E1)
        return E0, U0, E1, U1, E2, U2
    else:
        raise NotImplementedError("三阶及以上微扰")


if __name__ == "__main__":
    L = 10
    Hmat = sum(
        XX(i, (i + 1) % L) + YY(i, (i + 1) % L) + i * ZZ(i, (i + 1) % L)
        for i in range(L - 1)
    )
    Hmat.basis = spin_basis_1d(L=L)
    Hmat = Hmat.get_matrix()
    el, ev = Hmat.eigh()
    print(el)

    el, ev = eigh(Hmat, backend="quimb")
    print(el)
