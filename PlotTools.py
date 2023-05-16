import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg")

# from matplotlib.figure import Figure

config = {
    "pdf.fonttype": 42,
    "figure.dpi": 70,
    "font.size": 15,
    "axes.labelsize": 15,
    # "font.family": 'Times New Roman',
    "mathtext.fontset": "stix",  # 'dejavusans','dejavuserif', 'cm', 'stix','stixsans' or 'custom'
    "font.serif": ["SimSun"],
    "figure.autolayout": True,
    "xtick.direction": "in",  # x tick 方向
    "ytick.direction": "in",  # y tick 方向
    # grid
    "axes.grid": "False",
    "grid.alpha": 0.4,  # 透明度
    "grid.linewidth": 1.0,  # 粗细
    "svg.image_inline": True,
}
plt.rcParams.update(config)
# plt.savefig('',bbox_inches='tight')


def set_axis(ax, xlim, ylim, xlabel, ylabel):
    # ax, ax2 = fig.get_axes()  # 从 fig 获取 ax
    # ax.lines[0].set_label("Var")  # 给线添加 label
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # ax.legend()  # 显示 legend
    plt.tight_layout()
    # ax.set_xticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], fontsize=30)  # 该 x 方向 tick
    # ax.set_yticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], fontsize=30)  # 该 y 方向 tick
    # ax.tick_params(axis="both", which="both", direction='in')  # 选择齿方向 "in"/"out" which 可填 "major"/"minor"
    # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))  # x 轴上设置小齿间隔
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))  # y 轴上设置小齿间隔
    # ax.secondary_xaxis('top',
    #                    functions=(lambda x: np.interp(x, [-3, 3], [-4, 4]),
    #                               lambda x: np.interp(x, [-3, 3], [-4, 4])))  # 添加辅助轴
    # plt.savefig(r"data/GroundEnergy.pdf",bbox_inches="tight")


def find_boundary(x, y, zdata, a, clf=None, axes=None):
    """找到 (x, y, z) 图中的二分类边界，一边 z < a
    默认使用 Gassian 过程分类：clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    常用的分类器还有：
    # 支持向量机线性分类
    clf = sklearn.svm.SVC(kernel="linear", C=0.025)
    # 支持向量机分类
    clf = sklearn.gaussian_process.SVC(gamma=2, C=1)
    # 决策树分类
    clf = sklearn.tree.DecisionTreeClassifier(max_depth=5)
    # MLPC 分类
    clf = sklearn.neural_network.MLPClassifier(alpha=1, max_iter=1000)
    # 高斯朴素贝叶斯分类
    clf = sklearn.naive_bayes.GaussianNB()
    # 随机森林分类
    clf = sklearn.ensemble.RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    # AdaBoost 分类
    clf = sklearn.ensemble.AdaBoostClassifier()
    # 二次判别分析算法
    clf = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis()

    Args:
        x (numpy.ndarray): 1-d 数组，x 方向数据
        y (numpy.ndarray): 1-d 数组，y 方向数据
        zdata (numpy.ndarray): 2-d 数组，z 方向数据
        a (real): 分界线
        clf (classifier, optional): 分类器。Defaults to None.
        axes (list, optional): [x0, x1, y0, y1]. Defaults to None.

    Returns:
        (numpy.ndarray, numpy.ndarray): 边界的横纵坐标
    """
    from sklearn.metrics import accuracy_score

    assert np.all([isinstance(i, np.ndarray) for i in [x, y, zdata]])
    assert x.ndim == 1 and y.ndim == 1 and zdata.ndim == 2
    assert zdata.shape == (y.size, x.size)
    pts = np.array([[i, j] for i in x for j in y])
    vls = np.array([zdata[j, i] for i in range(len(x)) for j in range(len(y))]) < a
    if clf is None:
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF

        clf = GaussianProcessClassifier(1.0 * RBF(1.0))
    clf.fit(pts, vls)
    pre_y = clf.predict(pts)
    accuracy = accuracy_score(vls, pre_y)
    print(f"classify accuracy: {accuracy:0.2f}")
    if axes is None:
        axes = [x.min(), x.max(), y.min(), y.max()]
    x0s = np.linspace(axes[0], axes[1], 200)
    x1s = np.linspace(axes[2], axes[3], 200)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X)
    y_pred = y_pred.reshape(x0.shape)
    fig = plt.figure("temp")
    cs = fig.add_subplot(111).contour(x0, x1, y_pred, levels=[0.5])
    plt.close("temp")
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    return v[:, 0], v[:, 1]


def get_ydata(func, x_list, pars_dic=None, core_num=1):
    """equivilant to `[func(xi, **pars_dic) for xi in x_list]`
    but with multiprocess intergtated
    but multiprocess not necessary accelerate the process espeically with func is complicated, i.e., involving eigh

    Args:
        func (function): function to be run
        x_list (np.array/list): x data
        pars_dic (dictionary, optional): parameters func needs. Defaults to None.
        core_num (int, optional): the number of core used. Defaults to 1.

    Raises:
        Exception: lambda is not available when using multiprocess
        Exception: core should be less than the available number

    Returns:
        np.array: func(x_list)
    """
    if pars_dic is None:
        pars_dic = dict()
    if core_num == 1:
        y_list = [func(x_i, **pars_dic) for x_i in x_list]
    else:
        import multiprocessing

        if core_num > multiprocessing.cpu_count() - 2:
            raise Exception("核太多")
        from functools import partial

        if str(func)[:18] == "<function <lambda>":
            raise Exception("不支持使用 lambda 函数")
        mic_func_partial = partial(func, **pars_dic)
        with multiprocessing.Pool(processes=4) as pool:
            y_list = pool.map(mic_func_partial, x_list)
    return np.array(y_list)


def get_zdata(func, x_list, y_list, pars_dic=None):
    """equivilant to `[[f_(xi, yi) for xi in xdata] for yi in ydata]`，没写多核了

    Args:
        func (function): function to be run
        x_list (np.array/list): x data
        pars_dic (dictionary, optional): parameters func needs. Defaults to None.
        core_num (int, optional): the number of core used. Defaults to 1.

    Raises:
        Exception: lambda is not available when using multiprocess
        Exception: core should be less than the available number

    Returns:
        np.array: func(x_list)
    """
    if pars_dic is None:
        pars_dic = dict()
    z_list = [[func(x_i, y_i, **pars_dic) for x_i in x_list] for y_i in y_list]
    return np.array(z_list)


def direct_plot(func, x_list, pars_dict=None, ax=None, core_num=1):
    """equvilant to `plt.plot(x_list, [func(xi) for xi in x_list])`
    but with multiprocess intergtated
    but multiprocess not necessary accelerate the process espeically with func is complicated, i.e., involving eigh

    Args:
        func (function): function to be run
        x_list (np.array/list): x data
        pars_dic (dictionary, optional): parameters `func` needs. Defaults to None.
        ax (axes, optional): the axes that will be ploted on. Defaults to None.
        core_num (int, optional): the number of core used. Defaults to 1.

    Returns:
        y_list (list): y data gotten
        ax (axes): the axes plotted
    """
    if pars_dict is None:
        pars_dict = {}
    y_list = get_ydata(func, x_list, pars_dict, core_num)
    if ax is None:
        ax = plt.subplot(111)
    ax.plot(x_list, y_list)
    return y_list, ax


# def realtime_plot(func, x_list, pars_dict=None, ax=None, step_time=0.1, mk=""):
#     """dynamically plot the `func`,
#     it will slow down the process when `func` is simple enough
#     and stuck when `func` is too complicated while `step_time` is too small.
#     动图需要在 matplotlib qt 中进行，widget 目前没能实现这一功能

#     Args:
#         func (function): function to be plotted
#         x_list (list/np.array): x data
#         pars_dict (dictionary, optional): parameters `func` needs. Defaults to None.
#         ax (axes, optional): the axes that will be plotted on. Defaults to None.
#         step_time (float, optional): sleep time between each plot. Defaults to 0.1.

#     Returns:
#         y_list (list): y data gotten
#         lines (lines): the lines plotted
#         ax (axes): the axes plotted
#     """
#     if pars_dict is None:
#         pars_dict = {}
#     if ax is None:
#         ax = plt.subplot(111)
#     x_list_plot = []
#     y_list_plot = []
#     inilinenum = len(ax.lines)
#     for xi in x_list:
#         x_list_plot.append(xi)
#         yi = func(xi, **pars_dict)
#         y_list_plot.append(yi)
#         print(x_list_plot, y_list_plot)
#         lines = ax.plot(x_list_plot, y_list_plot, mk)
#         for i, line in enumerate(lines):
#             line.set_color('C%s' % (inilinenum+i))
#         points = ax.plot([xi], [yi], "o")
#         for i, point in enumerate(points):
#             point.set_color('C%s' % (inilinenum+i))
#         plt.pause(step_time)
#         for line in lines:
#             line.remove()
#         for point in points:
#             point.remove()
#     lines = ax.plot(x_list_plot, y_list_plot, mk)
#     for i, line in enumerate(lines):
#         line.set_color('C%s' % (inilinenum+i))
#     return y_list_plot, lines, ax


def realtime_plot(func, x_list, pars_dict=None, step_time=0.1, ax=None, lw="-"):
    """real_time_plot 的 widget 版本，需要在另一个 cell 中提前把画布建好

    Example:
    fig = plt.figure(figsize=(5,4))
    plt.clf()
    realtime_plot(lambda x: [np.cos(x), np.sin(x)], np.linspace(0, np.pi, 50), step_time=0.1)
    """
    if ax is None:
        ax = plt.subplot(111)
    if pars_dict is None:
        pars_dict = {}
    x_list_plot = []
    y_list_plot = []
    x0 = list(x_list).pop(0)
    x_list_plot.append(x0)
    y0 = func(x0, **pars_dict)
    try:
        _ = y0[0]
    except:
        y0 = [y0]
    y_list_plot.append(y0)
    lines = ax.plot(x_list_plot, y_list_plot, lw)
    points = ax.plot([x_list_plot[-1]], [y_list_plot[-1]], "o")
    for i in range(len(lines)):
        points[i].set_color(lines[i].get_color())
    xmin, xmax = x0, x0
    ymin, ymax = np.min(y0), np.max(y0)
    for xi in x_list:
        x_list_plot.append(xi)
        yi = func(xi, **pars_dict)
        if type(yi) != list:
            yi = [yi]
        y_list_plot.append(yi)
        for i, line in enumerate(lines):
            line.set_xdata(x_list_plot)
            line.set_ydata(np.array(y_list_plot)[:, i])
        for i, point in enumerate(points):
            point.set_xdata([x_list_plot[-1]])
            point.set_ydata([y_list_plot[-1][i]])
        if xi != xmin and xi != xmax:
            xmin, xmax = min([xi, xmin]), max([xi, xmax])
            ax.set_xlim([xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05])
        yimin, yimax = np.min(yi), np.max(yi)
        if yimin != ymin and yimax != ymax:
            ymin, ymax = np.min([ymin, yimin]), np.max([ymax, yimax])
            ax.set_ylim([ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05])
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        time.sleep(step_time)
    for point in points:
        point.remove()
    return y_list_plot


def animate_fig(x_lists, y_lists, ax=None, step_time=0.0, lw="-"):
    """将 x_list, y_list 以动画的形式放出来

    Args:
        x_lists (numpy.ndarray): 必须是 numpy 格式
        y_lists (numpy.ndarray): 必须是 numpy 格式，第二个指标表示不同的线
        ax (axes, optional): 轴。Defaults to None.
        step_time (float, optional): 休息时间。Defaults to 0.0.
    """
    if ax is None:
        ax = plt.subplot(111)
    tag = 1
    if x_lists.ndim == 1:
        x_lists = np.array(x_lists, ndmin=2).T
        tag = 0
    if y_lists.ndim == 1:
        y_lists = np.array(y_lists, ndmin=2).T
    lines = ax.plot([x_lists[0, :]], [y_lists[0, :]], lw)
    points = ax.plot([x_lists[0, :]], [y_lists[0, :]], "o")
    for i in range(len(lines)):
        points[i].set_color(lines[i].get_color())
    xmax, xmin = np.max(x_lists), np.min(x_lists)
    ax.set_xlim([xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05])
    ymax, ymin = np.max(y_lists), np.min(y_lists)
    ax.set_ylim([ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05])
    for l in range(x_lists.shape[0]):
        for i, line in enumerate(lines):
            line.set_xdata(x_lists[: l + 1, i * tag])
            line.set_ydata(y_lists[: l + 1, i])
        for i, point in enumerate(points):
            point.set_xdata([x_lists[l, i * tag]])
            point.set_ydata([y_lists[l, i]])
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        time.sleep(step_time)
    for point in points:
        point.remove()
    return ax


def addColorBar(
    fig,
    axes,
    mappable,
    orientation="vertical",
    location="right",
    pos_pars=None,
    ticks=None,
    tkd=None,
    label=None,
    tkl=None,
    tickdirection="out",
):
    """
    add a cax at the right of axes with the same height
    Args:
    mappable: 二维图返回的对象
    orientation(str): vertical/horizontal
    location(str): right/left/bottom/top
    pos_pars(list): x1, x2, y1, y2 = pos_pars 可以用 `axes.get_position()` 参考
    ticks(list): 要显示 ticks
    tkd(str): e.g. '{x:.3f}' the digits of the ticks
    label(str): 要显示的 label
    tkl(float): 小 ticks 的间距
    tickdirection(str): 指向里面还是指向外面.
    """
    ## 独立的 colorbar
    # if mappable is None:
    #     if cmap is None:
    #         raise Exception("需要 mappable 或者 cmap")
    #     cax, kwargs = mpl.colorbar.make_axes_gridspec(axes)
    #     cax.grid(visible=False, which='both', axis='both')
    #     im4 = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    #
    #     cbar = fig.colorbar(im4, cax=cax, orientation=orientation, location=location)
    #     NON_COLORBAR_KEYS = ['fraction', 'pad', 'shrink', 'aspect', 'anchor',
    #                          'panchor']
    #     cb_kw = {k: v for k, v in kwargs.items() if k not in NON_COLORBAR_KEYS}
    #     cb = mpl.colorbar.Colorbar(cax, im4, **cb_kw)
    # else:
    #     cbar = fig.colorbar(mappable, orientation=orientation,location=location)
    if pos_pars is None:
        cbar = fig.colorbar(mappable, orientation=orientation, location=location)
    else:
        x1, x2, y1, y2 = pos_pars
        caxpos = mpl.transforms.Bbox.from_extents(x1, y1, x2, y2)
        cax = axes.figure.add_axes(caxpos)
        cbar = fig.colorbar(
            mappable, cax=cax, orientation=orientation, location=location
        )
    cbar.minorticks_off()
    if ticks is not None:
        cbar.set_ticks(ticks)
        if label is not None:
            cbar.set_label(label)
        cbar.ax.tick_params(which="major", direction=tickdirection)
        if tkd is not None:
            cbar.formatter = mpl.ticker.StrMethodFormatter(tkd)
        if tkl is not None:
            cbar.minorticks_on()
            cbar.minorlocator = mpl.ticker.MultipleLocator(tkl)
            cbar.ax.tick_params(which="minor", direction=tickdirection)
        cbar.update_ticks()


def test_f(x):
    return [np.sin(x), np.cos(x)]

    # if __name__ == "__main__":
    # """find boundry"""
    # xdata = np.linspace(-2, 2, 15)
    # ydata = np.linspace(0, 2, 20)
    # zdata = np.array([[i**2 + j**2 for i in xdata] for j in ydata])
    # rng = np.random.default_rng(42)
    # zdata += rng.normal(scale=0.5, size=zdata.shape)
    # ax = plt.subplot(111)
    # ax.pcolor(xdata, ydata, zdata)
    # bdx, bdy = find_boundary(xdata, ydata, zdata, a=0.5, axes=[-1.2, 1.2, 0, 1.25])
    # plt.plot(bdx, bdy, 'r-.')
    # """realtime plot"""
    # x_list = np.linspace(0, 2 * np.pi, 20)
    # realtime_plot(test_f, x_list, step_time=0.1, mk="o-")
    # plt.show()
    # end for
    """direct_plot - single core"""
    # import time
    # x_list = np.linspace(0, 2 * np.pi, 10000)
    # t = time.time()
    # direct_plot(test_f,x_list, {"phi":2})
    # print(time.time() - t)
    # plt.show()
    """direct_plot - multi core"""
    # x_list = np.linspace(0, 2 * np.pi, 10000)
    # myfuc2 = lambda x, phi: np.sin(x + phi)
    # t = time.time()
    # direct_plot(test_f, x_list, {"phi":2}, core_num=2)  # 不支持使用 lambda 函数多核
    # print(time.time() - t)
    # plt.show()
    """contourf, addcolorbar"""
    # xlist = np.linspace(-3.0, 3.0, 100)
    # X, Y = np.meshgrid(xlist, xlist)
    # Z = np.sqrt(X ** 2 + Y ** 2)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # bounds =[0, 0.1,1]
    # norm = mpl.colors.BoundaryNorm(bounds, len(bounds) - 1)
    # cp = ax.contourf(xlist, xlist, Z, cmap = "Blues")
    # addColorBar(cp, fig, ax)
    # plt.show()
    """hist2d"""
    # N_points = 100000
    # x = np.random.randn(N_points)
    # y = 4 * x + np.random.randn(100000) + 50
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # _, _, _, t = ax.hist2d(x, y,bins=100,norm=mpl.colors.LogNorm(),cmap="gray")
    # addColorBar(t, fig, ax)
    # plt.show()
