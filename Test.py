from PyLib import *

if __name__ == "__main__":
    """Example 1"""  # for basis
    print("===========================")
    print("Example 1")
    sb = spin_basis_1d(3)
    print("plain")
    show_basis(sb)
    # show_basis(sb, vrtlist=[True])
    print("fancy")
    show_basis(sb, tp="fancy")
    print("x")
    show_basis(sb, tp="x")
    print("y")
    show_basis(sb, tp="y")

    """ Example 3 """  # for basis
    print("===========================")
    print("Example 3")
    sb = spin_basis_1d(3, Nup=2)
    show_basis(sb)
    print("fancy")
    show_basis(sb, tp="fancy")

    """ Example 4 """  # for basis
    print("===========================")
    print("Example 4")
    bb = boson_basis_1d(L=2, sps=5)
    show_basis(bb)
    show_basis(bb, vrtlist=[True])

    """ Example 5 """  # for basis
    print("===========================")
    print("Example 5")
    ecsb = spin_basis_1d(2, 1)
    bb = boson_basis_1d(2, sps=3)
    mb = tensor_basis(ecsb, bb)
    print(mb)
    show_basis(mb)

    """ Example 6 """  # for state
    print("===========================")
    print("Example 6")
    sb = spin_basis_1d(2)
    s = np.array([[0, 1 / 2, -1 / 2, 0]]).transpose()
    print(s)
    show_state(sb, s, tp="plain")  # 保留到 3 位小数

    """ Example 7 """  # for state
    print("===========================")
    print("Example 7")
    sb = spin_basis_1d(2)
    s1 = to_vector(sb, ["01", "10"], [1, 1])
    s2 = to_vector(sb, ["01", "11"], [-1, 1])
    s3 = s1 - s2
    show_state(sb, s1, 3)
    show_state(sb, s2, 3)
    show_state(sb, s3, 3)
    print(s1 @ s2)

    """ Example 8 """  # for state
    print("===========================")
    print("Example 8")
    bb = boson_basis_1d(3, sps=4)
    s1 = to_vector(bb, ["023", "001"], [1, 2])
    s2 = to_vector(bb, ["001"], [-1])
    s3 = s1 - s2
    show_state(bb, s1)
    show_state(bb, s2)
    show_state(bb, s3)
    print(s1 @ s2)

    # """ Example 9 """  # for state
    print("===========================")
    print("Example 9")
    sb = spin_basis_1d(1)
    bb = boson_basis_1d(1, sps=5)
    mb = tensor_basis(sb, bb)
    s1 = to_vector(mb, (("1", "1"),), (0.5,))
    s2 = to_vector(mb, (("1", "1"), ("0", "2")), (1, 0.3))
    s3 = s1 + s2
    show_state(mb, s1)
    show_state(mb, s2)
    show_state(mb, s3)

    """ Example 11 """  # for state
    print("===========================")
    print("Example 11")
    sb = spin_basis_1d(1)
    bb = boson_basis_1d(2, sps=10)
    mb = tensor_basis(sb, bb)
    s1 = to_vector(sb, ["0", "1"], [0.3, 0.9])
    s2 = to_vector(bb, ["10", "01"], [0.5, 0.3])
    s3 = np.kron(s1, s2)
    show_state(sb, s1, tp="plain")
    show_state(bb, s2, tp="plain")
    show_state(mb, s3, tp="plain")

    """ Example 12 """  # for oper repre
    """#  H_data = [ (,0.4), (,0.3), (,0.5) ]
    表示三个算符相加
    H_data = [ ( [(), (), ()], 0.4), ( [(), (), ()], 0.3) ]
    每项里面有三种类型基矢的成绩
    H_data = [ ( [(a1, b1), (c1, d1, e1), (f1, )], 0.4), ( [(a2), (b2,c2), (d2, e2, f2)], 0.3) ]
    每种基矢里面有几项的乘积，这里就表示 (a1 b1) (c1 d1 e1) (f1) + (a2) (b2 c2) (d2 e2 f2)
    其中每个 a1 = (oper, par)"""
    print("===========================")
    print("Example 12")
    L = 3
    static = [
        ["xx", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["yy", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["zz", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["z", [[1, 1]]],
    ]
    basis = spin_basis_1d(L=3, pauli=-1)
    Hqs = hamiltonian(
        static, [], basis=basis, check_herm=False, check_pcon=False, check_symm=False
    )

    H = Oper(static, ["s"])
    H.show_oper()

    H = sum(
        XX(i, (i + 1) % L) + YY(i, (i + 1) % L) + ZZ(i, (i + 1) % L)
        for i in range(L - 1)
    ) + Z(1)
    H.get_matrix(basis)
    H.show()

    """ Example 13 """  # for oper repre
    print("===========================")
    print("Example 13")
    L = 3
    static = [
        ["+-", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["-+", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["nn", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["z", [[1, 1]]],
    ]
    H = Oper(static, ["f"])
    H.show()
    H = sum(
        PM_f(i, (i + 1) % L) + MP_f(i, (i + 1) % L) + NN_f(i, (i + 1) % L)
        for i in range(L - 1)
    ) + Z_f(1)
    H.show()

    # """ Example 14 """  # for oper repre
    print("===========================")
    print("Example 14")
    L = 3
    static = [
        ["+-", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["-+", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["nn", [[1, i, (i + 1) % L] for i in range(L - 1)]],
        ["n", [[1, 1]]],
    ]
    H = Oper(static, ["b"])
    H.show()
    H = sum(
        MP_b(i, (i + 1) % L) + PM_b(i, (i + 1) % L) + NN_b(i, (i + 1) % L)
        for i in range(L - 1)
    ) + N_b(1)
    H.show()

    """ Example 15 """  # for oper repre
    print("===========================")
    print("Example 15")
    H = 1 / 2 * Z(0) + N_b(0) + ((SP(0) + SM(0)) @ (B(0) + Bdag(0)))  # Rabi
    H.show()
    H = 1 / 2 * Z(0) + N_b(0) + 0.1 * SP(0) @ B(0) + 0.1 * SM(0) @ Bdag(0)  # J-C
    H.show()

    """ Example 16 """  # for oper action
    print("===========================")
    print("Example 16")
    H = MP(1, 0) + PM(0, 1) + ZZ(1, 0) + 4 * Z(1) + 7 * SP(1) + 4 * SM(1)  # type: Oper
    H.show()
    sb = spin_basis_1d(2)
    s1 = to_vector(sb, ["01", "10"], [0.1, 0.3])
    show_state(sb, s1)

    H1 = H.get_LinearOperator(sb)
    s2 = H1.dot(s1)
    show_state(sb, s2)

    H2 = H.get_matrix(sb)
    s2 = H2.dot(s1)
    show_state(sb, s2)

    print(s2)

    """ Example 17 """  # for oper action
    print("===========================")
    print("Example 17")
    H = (
        2 * PM_b(0, 1)
        + 2 * MP_b(0, 1)
        + 4 * NN_b(0, 1)
        + 4 * N_b(1)
        + 7 * Bdag(1)
        + 4 * B(0)
    )  # type: Oper
    H.show_oper()
    sb = boson_basis_1d(2, sps=4)
    s1 = to_vector(sb, ["1", "2"], [1, 1])
    H = H.get_LinearOperator(sb)
    show_state(sb, s1)
    H.dot(s1)

    """ Example 18 """  # for oper action
    print("===========================")
    print("Example 18")

    from PyLib import *

    H = SP(0) @ B(0) + SM(0) @ Bdag(0)
    print(H.static)
    sb = spin_basis_1d(1)
    bb = boson_basis_1d(1, sps=3)
    mb = tensor_basis(bb, sb)
    s1 = to_vector(sb, ["0", "1"], [0.3, 0.9])
    s2 = to_vector(bb, ["1", "2"], [0.5, 0.3])
    s3 = np.kron(s1, s2)
    show_state(sb, s1)
    show_state(bb, s2)
    show_state(mb, s3)

    H1 = H.get_matrix(basis=mb, check_pcon=False)
    s4 = H1.dot(s3)
    show_state(mb, s4)

    """ Example 19 """  # for oper mat
    print("===========================")
    print("Example 19")
    site_num = 3
    H = (
        sum(MP(i, i + 1) + PM(i + 1, i) + ZZ(i, i + 1) for i in range(site_num - 1))
        + MP(0, site_num - 1)
        + MP(site_num - 1, 0)
        + ZZ(0, site_num - 1)
    )
    H.show_oper()
    sb = spin_basis_1d(site_num)
    H.get_matrix(basis=sb)

    """ Example 20 """
    print("===========================")
    print("Example 20")
    mode_num = 3
    cut_off = 2
    # """第一种方法"""
    H_data = [
        ["+-", [[1, i, i + 1] for i in range(mode_num - 1)]],
        ["+-", [[1, i + 1, i] for i in range(mode_num - 1)]],
        ["+-", [[1, mode_num - 1, 0]]],
        ["+-", [[1, 0, mode_num - 1]]],
    ]
    H = Oper(H_data, ["b"])
    H.show_oper()
    # """第二种方法"""
    H = (
        sum(PM_b(i, i + 1) + PM_b(i + 1, i) for i in range(mode_num - 1))
        + PM_b(mode_num - 1, 0)
        + PM_b(0, mode_num - 1)
    )
    H.show_oper()
    sb = boson_basis_1d(mode_num, sps=cut_off)
    s1 = to_vector(sb, ["101"], [1])
    show_state(sb, s1)

    H1 = H.get_matrix(basis=sb)
    s2 = H1.dot(s1)
    show_state(sb, s2)

    """ Example 21 """
    print("===========================")
    print("Example 21")
    mode_num = 3
    cut_off = 2
    H_data = [
        ["+-", [[1, i, i + 1] for i in range(mode_num - 1)]],
        ["+-", [[1, i + 1, i] for i in range(mode_num - 1)]],
        ["+-", [[1, mode_num - 1, 0]]],
        ["+-", [[1, 0, mode_num - 1]]],
    ]
    H = Oper(H_data, ["b"])
    H.show_oper()
    sb = boson_basis_1d(mode_num, sps=cut_off)
    show_basis(sb)
    H.get_matrix(basis=sb)

    #     """ Example 22 """
    #     print("===========================")
    print("Example 22")
    H = N_b(0) + N(0) + 0.1 * B(0) @ SP(0) + 0.1 * Bdag(0) @ SM(0)  # J-C
    H.show_oper()
    sb = spin_basis_1d(1, pauli=0)
    bb = boson_basis_1d(1, sps=4)
    mb = tensor_basis(bb, sb)
    show_basis(mb)
    H.get_matrix(basis=mb)
    H.get_matrix(s_par=1, b_par=[1, 4], pauli=-1)

    """ Example 23 """
    print("===========================")
    print("Example 23")
    H = N_b(0) + N(0) + 0.1 * B(0) @ SP(0) + 0.1 * Bdag(0) @ SM(0)  # J-C
    H.show_oper()
    sb = spin_basis_1d(1, pauli=-1)
    bb = boson_basis_1d(1, sps=4)
    mb = tensor_basis(bb, sb)
    H1 = H.get_matrix(mb)
    H1.eigh()
    # H.show_matrix(2)
    # H.get_eig()
    # H.show_eig()

    H2 = H.get_matrix(s_par=1, b_par=[1, 4], pauli=-1)
    H2.eigh()

    """ Example 24 """
    print("===========================")
    print("Example 24")
    site_num = 10
    H = (
        sum(MP(i, i + 1) + PM(i, i + 1) + ZZ(i, i + 1) for i in range(site_num - 1))
        + MP(0, site_num - 1)
        + PM(site_num - 1, 0)
        + ZZ(0, site_num - 1)
    )
    H.show_oper()
    H.get_matrix(s_par=site_num)
    sb = spin_basis_1d(site_num)
    H.get_matrix(sb)

    """ Example 26 """
    print("===========================")
    print("Example 26")
    site_num = 3
    H = (
        sum(
            PM_f(i, i + 1) + PM_f(i + 1, i) + ZZ_f(i, i + 1)
            for i in range(site_num - 1)
        )
        + PM_f(0, site_num - 1)
        + PM_f(site_num - 1, 0)
        + ZZ_f(0, site_num - 1)
    )
    H.show_oper()
    H.get_matrix(f_par=[site_num], pauli=0)  # 这个是 JW 变换 之后的矩阵
    el, ev = eigh(H1, backend="quimb")
    show_eig(H.basis, el, ev)

    sb = spinless_fermion_basis_1d(site_num)
    H2 = H.get_matrix(sb)  # 这个是在自然基上写的 我这里的自然基是指按 ...fdag_i...fdag_j |0> 中 i<j 排列的
    el, ev = eigh(H1, backend="quimb")
    show_eig(H.basis, el, ev)

    """ Example 27 """
    print("===========================")
    print("Example 27")
    sb = spin_basis_1d(1)
    bb = boson_basis_1d(3, sps=6)
    mb = tensor_basis(bb, sb)
    s = to_vector(mb, [("213", "0")], [1])
    show_state(mb, s)

    """ Example 33 """
    print("===========================")
    print("Example 33")
    H1 = XX(0, 1)
    H1.basis = spin_basis_1d(L=2)
    H1.show()
    H1.expand().show()
    H1.show_oper()

    """ Example 34 """
    print("===========================")
    print("Example 34")
    sb = spin_basis_1d(2, pauli=-1)
    s = to_vector(sb, ["11"], [1])
    show_state(sb, s)
    show_in_another_spin_coord(sb, s, "x")
