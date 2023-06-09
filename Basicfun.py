import scipy.io as sio
import numpy as _np
import os
# import torch as _tc

def clear(name):
    del(name)
    import gc
    gc.collect()

def creat_if_not_exist(parentpath, childepath):
    wholepath = parentpath + childepath
    if not os.path.exists(wholepath):
        os.makedirs(wholepath)
    return wholepath

# def choose_device(n=0):
#     """判断机器中的 cuda 是否可用"""
#     if n == "cpu":
#         return "cpu"
#     else:
#         if _tc.cuda.is_available():
#             if n is None:
#                 return _tc.device("cuda:0")
#             elif type(n) is int:
#                 return _tc.device("cuda:" + str(n))
#             else:
#                 return _tc.device("cuda" + str(n)[4:])
#         else:
#             return _tc.device("cpu")


def save_mat(path, H):
    H = H.tocsr().tocoo()
    H_coo = _np.zeros((3, len(H.row)))
    H_coo[0, :] = H.row
    H_coo[1, :] = H.col
    H_coo[2, :] = H.data
    sio.savemat(path, {'H_coo': H_coo, 'dim': H.shape[0]})


def save_hdf5(path, data):
    # save data to HDF5
    """ 保存成有结构的
    import h5py
    f = h5py.File("test.h5","a")
    f.create_group("test") # 创建一个 group
    g = f["test"]  # 记录 group 为一个变量，如果不存在该 "group" 会报错
    g.name  # 查看当前的所在文件目录位置
    list(g.keys()) # 查看当前的目录下的数据
    g["data"] = data # 将 data 储存进 "/test/data" 中，如果存在该数据会报错
    g["data"][1:2,:]  # 调用 "data" 中储存数据的第 1 到 2 行
    del g["data"]  # 删除 "/test/data" 中的数据
    f.close()  # 关闭文件
    """
    import h5py
    if path[-3:] != ".h5" and path[-5:] != ".hdf5":
        path += ".h5"
    with h5py.File(path+".h5", "w") as file:
        file.create_dataset("data", data=data)


def load_hdf5(path, ini=None):
    # read data to HDF5
    import h5py
    if path[-3:] != ".h5" and path[-5:] != ".hdf5":
        path += ".h5"
    if ini is None:
        with h5py.File(path, "r") as file:
            ini = file["data"][:]  # returns as a numpy array
    else:
        with h5py.File(path, "r") as file:
            file["data"].read_direct(ini)
    return ini


def martix_O_obs(psi, O):
    return psi.transpose().conj() @ O.dot(psi)


def Gauss_fun(mu, sigma2, xlist):
    Gauss = 1/(2*_np.pi*sigma2)**0.5 * _np.exp(-(xlist - mu)**2/(2*sigma2))
    return Gauss


def log_Gauss_fun(mu, sigma2, xlist):
    ylist = []
    for xi in xlist:
        yi = 1/(2*_np.pi*sigma2)**0.5 * _np.exp(-(_np.log(abs(xi)) - mu)**2/(2*sigma2)) / abs(xi) / 2
        ylist.append(yi)
    return ylist


def selection_Inx(array, left, right, inclu=True):
    if inclu:
        return (left <= array) * (array <= right)
    else:
        return (left < array) * (array < right)

def interp(x, y, x0, kind="linear"):
    """插值

    Args:
        x (list): x
        y (list): y
        x0 (list): x0
        kind (string, optional): 插值类型。Defaults to 'linear'.

    Returns:
        list: f(x0)
    """
    from scipy.interpolate import interp1d

    return interp1d(x, y, kind=kind, bounds_error=False)(x0)


def find_boundary(x, y, zdata, a, clf=None, axes=None):
    """找到 (x, y, z) 图中的二分类边界，一边 z < a
    默认使用 Gassian 过程分类：clf = GaussianProcessClassifier(1.0 * RBF(1.0))

    常用的分类器还有：
    # 支持向量机线性分类
    clf = sklearn.svm.SVC(kernel="linear", C=0.025)
    # 支持向量机分类
    clf = sklearn.svm.SVC(gamma=2, C=1)
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
    import matplotlib.pyplot as _plt

    assert _np.all([isinstance(i, _np.ndarray) for i in [x, y, zdata]])
    assert x.ndim == 1 and y.ndim == 1 and zdata.ndim == 2
    assert zdata.shape == (y.size, x.size)
    pts = _np.array([[i, j] for i in x for j in y])
    vls = _np.array([zdata[j, i] for i in range(len(x)) for j in range(len(y))]) < a
    if clf is None:
        from sklearn.svm import SVC

        clf = SVC(gamma="scale", C=10)
    clf.fit(pts, vls)
    pre_y = clf.predict(pts)
    accuracy = accuracy_score(vls, pre_y)
    print(f"classify accuracy: {accuracy:0.2f}")
    if axes is None:
        axes = [x.min(), x.max(), y.min(), y.max()]
    x0s = _np.linspace(axes[0], axes[1], 2000)
    x1s = _np.linspace(axes[2], axes[3], 2000)
    x0, x1 = _np.meshgrid(x0s, x1s)
    X = _np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X)
    y_pred = y_pred.reshape(x0.shape)
    fig = _plt.figure("temp")
    cs = fig.add_subplot(111).contour(x0, x1, y_pred, levels=[0.5])
    _plt.close("temp")
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    return v[:, 0], v[:, 1]

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
