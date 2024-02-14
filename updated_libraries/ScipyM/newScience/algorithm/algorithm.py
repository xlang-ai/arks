from typing import Any, Tuple, Union, Callable
import numpy as np


def two_sample_ks(x1: Any,
                  x2: Any,
                  choice: str = 'two-sided',
                  mode: str = 'auto'):
    import scipy.stats as st
    result = st.ks_2samp(x1, x2, choice, mode)
    return result


def dist_mat(x, y, metric: str = 'euclidean', p=2):
    import scipy.spatial as sp 
    if metric != 'minkowski':
        result = sp.distance.cdist(x, y, metric)
    else: 
        result = sp.distance.cdist(x, y, metric, p=p)
    return result


def one_sample_ks(x1: Any,
                  ref: Union[str, Callable],
                  args: Tuple = (),
                  choice: str = 'two-sided',
                  mode: str = 'auto'):
    import scipy.stats as st
    statistic, pvalue = st.ks_1samp(x1, ref, args, choice, mode)
    return (statistic, pvalue)


def best_2d_assign(cost, maximize: bool = False):
    import scipy.optimize as opt
    row_ind, col_ind = opt.linear_sum_assignment(cost, maximize)
    return (row_ind, col_ind)


def detect_features(array, connected_mode):
    import scipy.ndimage as image
    result = image.label(array, connected_mode)
    return result


def group_sum(array, labels=None, group=None):
    import scipy.ndimage as image
    result = image.sum_labels(array, labels, group)
    return result


def block_mat_with_diag(matrices):
    import scipy.linalg as lin
    result = lin.block_diag(*matrices)
    return result


def discretre_cos_trans(array, type: int = 2, length=None,
                        axis: int = -1,
                        norm=None):
    import scipy
    result = scipy.fft.dct(array, type, n=length, axis=axis, norm=norm)
    return result


def resize(src, dst_size, spline_factor=1):
    import scipy.ndimage as image
    ori_size = src.shape
    zoom_factor = [dst_size[i] / ori_size[i] for i in range(len(ori_size))]
    x = image.zoom(src, zoom_factor, order=spline_factor)
    return x


def root_digger(func, initial_guess: np.ndarray, args=()):
    import scipy.optimize as opt
    result = opt.fsolve(func, initial_guess, args)

    return result


def relative_argextreme(array,
                        comp_func,
                        axis=-1,
                        comp_num=1) -> Tuple[np.ndarray]:
    from scipy.signal import argrelextrema

    result = argrelextrema(array, comp_func, axis, comp_num)
    return result


def knn(center,
        search_scope,
        k=1,
        p=2, ):
    import scipy.spatial
    kdtree = scipy.spatial.cKDTree(search_scope)
    distances, indices = kdtree.query(center, k, p=p)
    return distances, indices


def rt_array(
        src,
        angle,
        resize: bool = True,
        axes: Tuple[int] = (1, 0)
):
    import scipy.ndimage as ndimage
    from math import degrees
    degree_angle = degrees(angle)

    rotated_image = ndimage.rotate(src, degree_angle, axes=axes, reshape=resize)
    return rotated_image


def normalize(src, method, axis=None):
    from scipy import stats
    if method == "zscore":
        result = stats.zscore(src, axis)
    elif method == "minmax":
        src = np.asarray(src)
        result = (src - np.min(src, axis=axis)) / (np.max(src, axis=axis) - np.min(src, axis=axis))
    else:
        raise ValueError("Not supported method type!")
    return result


def medianFilter(src, filter_shape, filter_kernel=None,
                 padding_mode="reflect", shift=0.0):
    from scipy import ndimage
    origin = (-np.array(shift)).tolist()

    result = ndimage.median_filter(src, filter_shape, filter_kernel, mode=padding_mode, origin=origin)
    return result


from scipy.spatial import Voronoi


class VoronoiPartition(Voronoi):
    def __init__(self, points, **kwargs) -> None:
        super().__init__(points, **kwargs)


def lineSearch(func,
               grad,
               initial_point,
               initial_dire,
               return_alpha=True):
    from scipy import optimize
    result = optimize.line_search(func, grad, initial_point, initial_dire)
    if return_alpha:
        return result[0]
    else:
        return result

def wilcoxon_rktest(
    sample1,
    sample2,
    alternative=None
):
    from scipy.stats import ranksums
    result = ranksums(sample1, sample2, alternative if alternative is not None else 'two-sided')
    return result

def pearson_kurtosis(src,
                     axis=None, bias=True):
    import scipy.stats 
    result = scipy.stats.kurtosis(src, axis, fisher=False, bias=bias)
    return result

def discrete_cos_trans(
    array, 
    type: int = 2, 
    length: Any  = None, 
    axis: int = -1, 
    norm: Any  = None,  
):
    import scipy.fft 
    result = scipy.fft.dct(array, type, length, axis, norm)
    return result