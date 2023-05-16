################
# 简单的，随处可见的小程序
###############

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
