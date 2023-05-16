# Time     : 2022/10/8 11:20
# Author   : Dingzu Wang
# FileName : 2014
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import math
from scipy.stats import norm

def energy_density(eigenvalues, ax=None, bandwidth=1.):
    """利用 KDE 方法快速画出连续的能级密度分布曲线，调节 bandwidth 会改变结果
    """
    tag = False
    color1 = "C0"  # 统计图的颜色
    color2 = "|C1"  # 能级的线和颜色
    if ax is None:
        ax = plt.subplot(111)
        tag = True
    x_d = np.linspace(eigenvalues.min(),eigenvalues.max(),100)
    density = sum(norm(xi,bandwidth).pdf(x_d) for xi in eigenvalues)/len(eigenvalues)
    ax.fill_between(x_d, density, color=color1, alpha=0.8)
    ax.plot(eigenvalues, np.full_like(eigenvalues, -0.002),color2,markeredgewidth=0.1,markersize=10)
    if tag:
        plt.show()
    return ax

def energy_hist(val, ax=None, bins=None):
    """利用 KDE 方法快速画出连续的能级密度分布曲线，调节 bandwidth 会改变结果
    """
    tag = False
    color1 = "C0"  # 统计图的颜色
    color2 = "|C1"  # 能级的线和颜色
    if ax is None:
        ax = plt.subplot(111)
        tag = True
    if bins is None:
        h = 1.05*np.std(val) * val.size**(-1/5) # 这是对高斯分布最优的选择，其它分布也应当保证 N**(-1/5)
        bins=np.arange(val.min(), val.max()+h,h)
    ax.hist(val, bins=bins, density=True, color=color1)
    ax.plot(val, np.full_like(val, -0.002),color2,markeredgewidth=0.1,markersize=10)
    if tag:
        plt.show()
    return ax

def unfolding(val, discard=0.1, polynomial_of_degree=15):
    ### cdf
    E_list, NE_list = [], []  # unit step function Theta: less or equal
    for i in range(len(val)-1):
        E_list.append(val[i]); NE_list.append(i)
        E_list.append(val[i]); NE_list.append(i+1)
    ### unfolding
    Fit = np.polynomial.Polynomial.fit(E_list, NE_list, polynomial_of_degree)  # polynomial fitting - degree 15
    val_discard = val[int(len(val)*discard):-int(len(val)*discard)]  # discard the spectrum located at the edges
    eps = Fit(np.array(val_discard))  # unfolded energy
    assert (eps[-1]-eps[0])/(len(eps)-1)-1. < 1.e-2  # mean level density = mean level spacing = 1
    return eps

def level_spacing_distribution(eps, ax=None, bins=None):
    """ 能级间距分布 """
    tag = False
    if ax is None:
        ax = plt.subplot(111)
        tag = True
    eps_spc = np.diff(eps)
    ### stats
    if bins is None:
        h = 1.05*np.std(eps_spc) * eps_spc.size**(-1/5) # 这是对高斯分布最优的选择，其它分布也应当保证 N**(-1/5)
        bins=np.arange(eps_spc.min(), eps_spc.max()+h,h)
    else:
        bins=np.linspace(0, 4+0.1,bins)
    ax.hist(eps_spc, bins=bins, density=True, color='lightgray', ec="gray")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ### comparing
    sn = np.linspace(0, 4, 100)
    Ps_poisson = np.exp(-sn)
    Ps_WD = np.pi * sn / 2 * np.exp(-np.pi * sn ** 2 / 4)
    ax.plot(sn, Ps_poisson, color="red", label='poisson')
    ax.plot(sn, Ps_WD, color="blue", label='WD')
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$P$")
    ax.legend()
    if tag:
        plt.show()
    return ax

def level_spacing_indicator_eta(eps):
    """ eta 指数 可积：1 不可积：0 """
    eps_spc = np.diff(eps)
    ### stats
    s0 = 0.4729
    eps_spc.sort()
    integral_Ps = np.count_nonzero(eps_spc<s0)/eps_spc.size
    integral_Pp_s = 0.37680761269947016
    integral_PWD_s = 0.16108178372342252
    eta = (integral_Ps - integral_PWD_s) / (integral_Pp_s - integral_PWD_s)
    return eta

def peak_position(eps, polynomial_of_degree=15):
    """ 峰值位置 可积：0 不可积：0.8 """
    eps_spc = np.diff(eps)
    eps_spc.sort()
    e_list, Ne_list = [], []  # unit step function Theta: less or equal
    for i in range(len(eps_spc)-1):
        e_list.append(eps_spc[i]); Ne_list.append(i)
        e_list.append(eps_spc[i]); Ne_list.append(i+1)
    e_list = np.array(e_list)
    Ne_list = np.array(Ne_list)/Ne_list[-1]
    fit = np.polynomial.Polynomial.fit(e_list, Ne_list, polynomial_of_degree).deriv()  # polynomial fitting - degree 15
    rt = fit.deriv().roots()
    rt = rt[(abs(rt.imag)<1e-10)*(rt.real>=0)*(rt.real<=1)].real
    rt = np.append(rt, [eps_spc[0]]) 
    Pk = rt[np.argmax(fit(rt))]
    return Pk

def level_number_variance(eps, l_list, ax=None):
    """ Sigma """
    tag = False
    if ax is None:
        ax = plt.subplot(111)
        tag = True
    Sigma_l = []
    for l in l_list:
        if l==0:
            Sigma_l.append(0)
            continue
        N_eps, _ = np.histogram(eps, np.arange(eps[0], eps[-1]+0.1, l), density=False)
        Sigma_l.append(np.var(N_eps))
    ax.plot(l_list, Sigma_l)
    ax.plot(l_list, l_list, 'b-.')
    y_d = [2*(np.log(2*np.pi*l)+np.euler_gamma+1-np.pi**2/8)/np.pi**2 if l!=0 else 0 for l in l_list]
    ax.plot(l_list, y_d, 'r--')
    ax.set_ylim([0,2])
    if tag:
        plt.show()
    return l_list, ax

def level_spacing_indicator_beta(eps, bandwidth=0.05):
    """ beta 指数 可积：0 不可积：1 """
    ### fit eps with poly
    eps_spc = np.diff(eps)
    b = lambda beta: math.gamma((beta + 2)/(beta + 1))**(beta + 1)
    PB = lambda s, beta: b(beta) * (beta+1) * s**beta * np.exp( - b(beta) * s**(beta+1))
    s_order = np.linspace(0, np.max(eps_spc), 1000)[1:]
    Ps = sum(norm(xi,bandwidth).pdf(s_order) for xi in eps_spc)/len(eps_spc)  # 有参数可调
    beta = optimize.curve_fit(PB, s_order, Ps)[0][0]
    return beta


def ratio_of_adjacent_level_spacings(val, ax=None, bins=None):
    """ r 的分布 """
    if ax is None:
        ax = plt.subplot(111)
    s_list = np.diff(val)
    r_list = []
    for i in range(len(s_list)-1):
        if s_list[i] < 1e-10 or s_list[i+1] < 1e-10:
            r_list.append(0)
        else:
            r = min(s_list[i]/s_list[i+1], s_list[i+1]/s_list[i])
            r_list.append(r)
    if bins is None:
        h = 1.05*np.std(r_list) * len(r_list)**(-1/5) # 这是对高斯分布最优的选择，其它分布也应当保证 N**(-1/5)
        bins=np.arange(np.min(r_list), np.max(r_list)+h,h)
    else:
        bins=np.linspace(0, 1+0.1,bins)
    ax.hist(r_list, bins=bins, density=True, color='lightgray', ec="gray")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 2)
    """RMT result comparing"""
    r = np.linspace(0, 1, 100)
    P_poisson = 2 / ( r + 1 )**2
    Z1 = 8/27
    P_GOE = 1/Z1 * 2 * (r + r**2) / ( 1 + r + r**2 )**(5/2)
    ax.plot(r, P_poisson, color="red", label='poisson')
    ax.plot(r, P_GOE, color="blue", label='WD')
    # 设置横纵坐标的名称以及对应字体格式
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 14}
    ax.set_xlabel(r"$r$", font)
    ax.set_ylabel(r"$P(r)$", font)
    ax.legend(prop={'size': 12}, loc='upper right')
    return ax

def ratio_avg(val):
    """ 平均 r 可积：0.38 混沌：0.53"""
    """same as quspin.tools.misc.mean_level_spacing"""
    s_list = np.diff(val)
    r_list = []
    for i in range(len(s_list)-1):
        if s_list[i] < 1e-10 or s_list[i+1] < 1e-10:
            r_list.append(0)
        else:
            r = min(s_list[i]/s_list[i+1], s_list[i+1]/s_list[i])
            r_list.append(r)
    return np.mean(r_list)
    

if __name__ == "__main__":
    from quspin.operators import hamiltonian, quantum_operator  # Hamiltonians and operators
    from quspin.basis import spin_basis_1d  # Hilbert space spin basis
    from quspin.tools.measurements import diag_ensemble
    L, Nup = 12, 6
    alpha = 1
    Delta = 0.5
    basis = spin_basis_1d(L, Nup, pauli=True)
    J_xy = [[alpha, i, i+1] for i in range(L-1)]
    J_zz = [[Delta, i, i+1] for i in range(L-1)]
    H_XXZ = [['xx', J_xy], ['yy', J_xy], ['zz', J_zz]]
    J_z_h = [[1., L//2]]
    sigmaZ_i = [['z', J_z_h]]
    J_z_b = [[1., i] for i in range(0, L-1, 2)]
    sigmaZ_odd = [['z', J_z_b]]
    operator_dict_SI = dict(H_XXZ=H_XXZ, sigmaZ_i=sigmaZ_i)
    H_block_SI = quantum_operator(operator_dict_SI, basis=basis, check_symm=False, check_pcon=False, check_herm=False)
    # fig = plt.figure(figsize=(8, 8))

    """level_spacing_distribution"""
    ### 可积
    h = 0.01
    param_dict = dict(H_XXZ=1., sigmaZ_i=h)
    H_SI = H_block_SI.tohamiltonian(param_dict)
    val = H_SI.eigvalsh()
    eps = unfolding(val, discard=0.1, polynomial_of_degree=15)
    # Pk = peak_position(eps)
    ax = plt.subplot(3, 2, 1)
    level_spacing_distribution(eps, ax=ax, bins=30)
    # ax = plt.subplot(3, 2, 3)
    # ratio_of_adjacent_level_spacings(val, ax=ax, bins=30)
    # ax = plt.subplot(3, 2, 5)
    # level_number_variance(eps, np.linspace(0,20,100), ax=ax)
    # eta = level_spacing_indicator_eta(eps)
    # Pk = peak_position(eps)
    # beta = level_spacing_indicator_beta(eps)
    # r_avg = ratio_avg(val)
    # print("eta=%.3f/1, Pk=%.3f/0, beta=%.3f/0, r_avg=%.3f/0.38" % (eta, Pk, beta, r_avg))
    
    ### 混沌
    h = 0.5
    param_dict = dict(H_XXZ=1., sigmaZ_i=h)
    H_SI = H_block_SI.tohamiltonian(param_dict)
    val = H_SI.eigvalsh()
    eps = unfolding(val, discard=0.1, polynomial_of_degree=15)
    ax = plt.subplot(3, 2, 2)
    level_spacing_distribution(eps, ax=ax, bins=30)
    # ax = plt.subplot(3, 2, 4)
    # ratio_of_adjacent_level_spacings(val, ax=ax, bins=30)
    # ax = plt.subplot(3, 2, 6)
    # level_number_variance(eps, np.linspace(0,20,100), ax=ax)
    # eta = level_spacing_indicator_eta(eps)
    # Pk = peak_position(eps)
    # beta = level_spacing_indicator_beta(eps)
    # r_avg = ratio_avg(val)
    # print("eta=%.3f/0, Pk=%.3f/0.8, beta=%.3f/1, r_avg=%.3f/0.53" % (eta, Pk, beta, r_avg))

    plt.show()
