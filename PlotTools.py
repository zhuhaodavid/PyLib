import numpy as _np
import matplotlib as _mpl
import matplotlib.pyplot as _plt

def set_axis(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _plt.tight_layout()
    ###############################################################
    # 画布、画框
    ###############################################################
    # fig = plt.figure(figsize=(6, 4)) # type:Figure
    # ax = fig.add_subplot(1, 1, 1)  # type:Axes

    # ax, ax2 = fig.get_axes()  # 从画布获取画框
    # [[x0, y0], [x1, y1]] = ax.get_position().get_points()  # 获取画框位置

    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)  # 画框间距

    # 添加 inset
    # ax_ = plt.axes(
    #     (x0 + 0.15 * (x1 - x0), y0 + 0.67 * (y1 - y0), 0.284 * (x1 - x0), 0.286 * (y1 - y0)))  # type:axes._axes.Axes


    ###############################################################
    # plot, pcolormesh
    ###############################################################
    # plot 参数：
    #   栅格化某一个线：plt.plot(...,  rasterized=True)
    #   点线边角设为圆角：line.set_dash_capstyle("round")

    # pcolormesh 参数：
    #   色彩范围：plt.pcolormesh(...,  norm=mpl.colors.Normalize(0.3863, 0.5307))
    #   栅格化：plt.pcolormesh(...,  rasterized=True)



    ###############################################################
    # label
    ###############################################################
    # ax.lines[0].set_label("Var")  # 给线添加 label
    # plt.legend(prop={'size': 14}, bbox_to_anchor=(0.7, 0.4))  # label 显示的位置和大小

    # 改变 label 顺序
    # handles, labels = ax.get_legend_handles_labels() 
    # order = [3, 2, 1, 0]
    # ax.legend([handles[i] for i in order], [labels[i] for i in order], prop={'size': 12}, loc="lower right", handlelength=3)



    ###############################################################
    # ticks
    ###############################################################
    # ax.set_xticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], fontsize=30)  # 改 x 方向 tick
    # ax.set_yticks([-3, -2, -1, 0, 1, 2, 3], [-3, -2, -1, 0, 1, 2, 3], fontsize=30)  # 改 y 方向 tick

    # ax.tick_params(axis="both", which="both", direction='in')  # 选择齿方向 "in"/"out" which 可填 "major"/"minor"

    # ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))  # x 轴上设置小齿间隔
    # ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.1))  # y 轴上设置小齿间隔

    # 增加辅助轴
    # ax__ = ax_.secondary_yaxis('right') 

    # ax__ = ax.secondary_xaxis('top',
    #                    functions=(lambda x: np.interp(x, [-3, 3], [-4, 4]),
    #                               lambda x: np.interp(x, [-3, 3], [-4, 4])))  # 添加辅助轴的映射关系

    # ax__.set_yticks([0.01, 0.1], [r"$10^{-2}$", r"$10^{-1}$"], fontsize=16)  # 该 y 方向 tick 添加辅助轴干掉左轴



    ###############################################################
    # save
    ###############################################################
    # plt.savefig(r"data/GroundEnergy.pdf",bbox_inches="tight", dpi=200)



def ini_mpl(config=None, reset=False, svg=True):
    # https://matplotlib.org/stable/tutorials/introductory/customizing.html
    if svg:
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats("svg")
    else:
        from IPython.display import set_matplotlib_formats
        set_matplotlib_formats("png")
    if config is None and reset is False:
        defaultconfig = {
            "pdf.fonttype": 42,
            "figure.dpi": 150,
            "font.size": 14,
            "axes.labelsize": 14,
            "font.family": 'Times New Roman', # 'sans-serif', "Times New Roman"
            "mathtext.fontset": "stix",  # 'dejavusans','dejavuserif', 'cm', 'stix','stixsans' or 'custom'
            "font.serif": ["SimSun"],
            # "figure.autolayout": True,
            "xtick.direction": "in",  # x tick 方向
            "ytick.direction": "in",  # y tick 方向
            # grid
            "axes.grid": "False",
            "grid.alpha": 0.4,  # 透明度
            "grid.linewidth": 1.0,  # 粗细
            # "svg.image_inline": True
        }
        _plt.rcParams.update(defaultconfig)
    if reset:
        config = {
            "pdf.fonttype": 3,
            "figure.dpi": 100,
            "font.size": 10.0,
            "axes.labelsize": "medium",
            "font.family": 'sans-serif', # 'sans-serif', "Times New Roman"
            "mathtext.fontset": "dejavusans",  # 'dejavusans','dejavuserif', 'cm', 'stix','stixsans' or 'custom'
            "font.serif": ["DejaVu Serif", "Bitstream Vera Serif", "Computer Modern Roman", "New Century Schoolbook", "Century Schoolbook L", "Utopia", "ITC Bookman", "Bookman, Nimbus Roman No9 L", "Times New Roman", "Times, Palatino", "Charter", "serif"],
            # "figure.autolayout": True,
            "xtick.direction": "out",  # x tick 方向
            "ytick.direction": "out",  # y tick 方向
            # grid
            "axes.grid": "False",
            "grid.alpha": 1.0,  # 透明度
            "grid.linewidth": 0.8,  # 粗细
            # "svg.image_inline": True
        }
        _plt.rcParams.update(config)
    if config is not None:
        _plt.rcParams.update(config)

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
        caxpos = _mpl.transforms.Bbox.from_extents(x1, y1, x2, y2)
        cax = axes.figure.add_axes(caxpos)
        cbar = fig.colorbar(mappable, cax=cax, orientation=orientation)
    cbar.minorticks_off()
    if ticks is not None:
        cbar.set_ticks(ticks)
        if label is not None:
            cbar.set_label(label)
        cbar.ax.tick_params(which="major", direction=tickdirection)
        if tkd is not None:
            cbar.formatter = _mpl.ticker.StrMethodFormatter(tkd)
        if tkl is not None:
            cbar.minorticks_on()
            cbar.minorlocator = _mpl.ticker.MultipleLocator(tkl)
            cbar.ax.tick_params(which="minor", direction=tickdirection)
        cbar.update_ticks()


def realtime_plot(func, x_list, pars_dict=None, step_time=0.1, ax=None, ls="-"):
    """real_time_plot 的 widget 版本，需要在另一个 cell 中提前把画布建好

    Example:
    fig = plt.figure(figsize=(5,4))
    plt.clf()
    realtime_plot(lambda x: [np.cos(x), np.sin(x)], np.linspace(0, np.pi, 50), step_time=0.1)
    """
    if pars_dict is None:
        pars_dict = {}
    if ax is None:
        ax = _plt.subplot(111)
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
    lines = ax.plot(x_list_plot, y_list_plot, ls)
    points = ax.plot([x_list_plot[-1]], [y_list_plot[-1]], "o")
    for i in range(len(lines)):
        points[i].set_color(lines[i].get_color())
    xmin, xmax = x0, x0
    ymin, ymax = _np.min(y0), _np.max(y0)
    for xi in x_list:
        x_list_plot.append(xi)
        yi = func(xi, **pars_dict)
        if type(yi) != list:
            yi = [yi]
        y_list_plot.append(yi)
        for i, line in enumerate(lines):
            line.set_xdata(x_list_plot)
            line.set_ydata(_np.array(y_list_plot)[:, i])
        for i, point in enumerate(points):
            point.set_xdata([x_list_plot[-1]])
            point.set_ydata([y_list_plot[-1][i]])
        if xi != xmin and xi != xmax:
            xmin, xmax = min([xi, xmin]), max([xi, xmax])
            ax.set_xlim([xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05])
        yimin, yimax = _np.min(yi), _np.max(yi)
        if yimin != ymin and yimax != ymax:
            ymin, ymax = _np.min([ymin, yimin]), _np.max([ymax, yimax])
            ax.set_ylim([ymin - (ymax - ymin) * 0.05, ymax + (ymax - ymin) * 0.05])
        ax.figure.canvas.draw()
        ax.figure.canvas.flush_events()
        _plt.pause(step_time)
    for point in points:
        point.remove()
    return y_list_plot


def animate_fig(xs, ys, ax=None, step_time=0.0, ls="-"):
    """将 x_list, y_list 以动画的形式放出来

    Args:
        x_lists (numpy.ndarray): 必须是 numpy 格式
        y_lists (numpy.ndarray): 必须是 numpy 格式，第二个指标表示不同的线
        ax (axes, optional): 轴。Defaults to None.
        step_time (float, optional): 休息时间。Defaults to 0.0.
    """
    x_lists, y_lists = _np.asarray(xs), _np.asarray(ys)
    if ax is None:
        ax = _plt.subplot(111)
    tag = 1
    if x_lists.ndim == 1:
        x_lists = _np.array(x_lists, ndmin=2).T
        tag = 0
    if y_lists.ndim == 1:
        y_lists = _np.array(y_lists, ndmin=2).T
    lines = ax.plot([x_lists[0, :]], [y_lists[0, :]], ls)
    points = ax.plot([x_lists[0, :]], [y_lists[0, :]], "o")
    for i in range(len(lines)):
        points[i].set_color(lines[i].get_color())
    xmax, xmin = _np.max(x_lists), _np.min(x_lists)
    ax.set_xlim([xmin - (xmax - xmin) * 0.05, xmax + (xmax - xmin) * 0.05])
    ymax, ymin = _np.max(y_lists), _np.min(y_lists)
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
        _plt.pause(step_time)
    for point in points:
        point.remove()
    return ax


def test_f(x):
    return [_np.sin(x), _np.cos(x)]


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
    """realtime plot"""
    # x_list = np.linspace(0, 2 * np.pi, 20)
    # fig = plt.figure(figsize=(5, 4))
    # realtime_plot(test_f, x_list, step_time=0.1, ls="o-")
    # plt.show()
    """animate fig"""
    # x_list = np.linspace(0, 2 * np.pi, 20)
    # y_list = [np.sin(xi) for xi in x_list]
    # animate_fig(x_list, y_list, ls="o-", step_time=0.1)
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
