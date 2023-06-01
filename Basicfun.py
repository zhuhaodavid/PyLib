import scipy.io as sio


def save_mat(path, H):
    H = H.tocsr().tocoo()
    H_coo = np.zeros((3, len(H.row)))
    H_coo[0, :] = H.row
    H_coo[1, :] = H.col
    H_coo[2, :] = H.data
    sio.savemat(path, {'H_coo': H_coo, 'dim': H.shape[0]})


def save_hdf5(path, data):
    # save data to HDF5
    import h5py
    with h5py.File(path+".h5", "w") as file:
        file.create_dataset("data", data=data)


def load_hdf5(path, int=None):
    # read data to HDF5
    import h5py
    if path[-3:] != ".h5":
        path += ".h5"
    if int is None:
        with h5py.File(path, "r") as file:
            int = file["data"][()]  # returns as a numpy array
    else:
        with h5py.File(path, "r") as file:
            file["data"].read_direct(int)
    return int


def martix_O_obs(psi, O):
    return psi.transpose().conj() @ O.dot(psi)


def Gauss_fun(mu, sigma2, xlist):
    Gauss = 1/(2*np.pi*sigma2)**0.5 * np.exp(-(xlist - mu)**2/(2*sigma2))
    return Gauss


def log_Gauss_fun(mu, sigma2, xlist):
    ylist = []
    for xi in xlist:
        yi = 1/(2*np.pi*sigma2)**0.5 * np.exp(-(np.log(abs(xi)) - mu)**2/(2*sigma2)) / abs(xi) / 2
        ylist.append(yi)
    return ylist


def selection_Inx(array, left, right, inclu=True):
    if inclu:
        return (left <= array) * (array <= right)
    else:
        return (left < array) * (array < right)





import torch as tc


def choose_device(n=0):
    """判断机器中的 cuda 是否可用"""
    if n == "cpu":
        return "cpu"
    else:
        if tc.cuda.is_available():
            if n is None:
                return tc.device("cuda:0")
            elif type(n) is int:
                return tc.device("cuda:" + str(n))
            else:
                return tc.device("cuda" + str(n)[4:])
        else:
            return tc.device("cpu")


def combine_dicts(dic_def, dic_new, deep_copy=False):
    # dic_def 中的重复 key 值将被 dic_new 覆盖
    import copy

    if dic_new is None:
        return dic_def
    if deep_copy:
        return dict(copy.deepcopy(dic_def), **copy.deepcopy(dic_new))
    else:
        return dict(dic_def, **dic_new)


def inverse_permutation(perm):
    """123->perm 的逆

    Args:
        perm (torch.tensor): 排序结果

    Returns:
        torch.tensor: 逆
    """
    if not isinstance(perm, tc.Tensor):
        perm = tc.tensor(perm)
    inv = tc.empty_like(perm)
    inv[perm] = tc.arange(perm.size(0), device=perm.device)
    return inv.tolist()


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
