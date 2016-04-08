import numpy as np
from ..utils import check_X_y

"""A brute-force method of kernel density estimation.
"""

######################################################################
# define the reverse mapping of VALID_METRICS
VALID_METRICS = ['EuclideanDistance']
from dist_metrics import get_valid_metric_ids
VALID_METRIC_IDS = get_valid_metric_ids(VALID_METRICS)


#########################################################
class BruteForceKDE:
    """
    """
    valid_metrics = VALID_METRIC_IDS

    def __init__(self, X, metric=None, leaf_size=None, weights=None, 
                 **kwargs):

        if weights is None:
            weights = np.ones(X.shape[0])

        self.data = X
        self.weights = weights
        pass

    def kernel_density(self, X, h, kernel='gaussian',
                       atol=0, rtol=0,
                       breadth_first = True, return_log=False):
        """
        kernel_density(self, X, h, kernel='gaussian', atol=0, rtol=1E-8,
                       breadth_first=True, return_log=False)

        Same syntax as for tree-based kernel_density (see binary_tree.pxi).  Some
        parameters are not used in this brute-force version.

        Compute the kernel density estimate at points X with the given kernel.

        Parameters
        ----------
        X : array_like
            An array of points to query.  Last dimension should match dimension
            of training data.
        h : float
            the bandwidth of the kernel
        kernel : string
            specify the kernel to use.  Options are
            - 'gaussian'
            - 'tophat'
            - 'epanechnikov'
            - 'exponential'
            - 'linear'
            - 'cosine'
            Default is kernel = 'gaussian'
        atol, rtol : float (default = 0)
            Specify the desired relative and absolute tolerance of the result.
            If the true result is K_true, then the returned result K_ret
            satisfies ``abs(K_true - K_ret) < atol + rtol * K_ret``
            The default is zero (i.e. machine precision) for both.
        breadth_first : boolean (default = False)
            if True, use a breadth-first search.  If False (default) use a
            depth-first search.  Breadth-first is generally faster for
            compact kernels and/or high tolerances.
        return_log : boolean (default = False)
            return the logarithm of the result.  This can be more accurate
            than returning the result itself for narrow kernels.

        """
        # validate kernel
        kernel_c = kernel_code(kernel)

        # number of points to query.
        n_query = X.shape[0]
        n_features = X.shape[1]
        
        if n_features != self.data.shape[1]:
            raise Exception("Mismatched number of features/dimensions")

        # Calculate normalization of kernel function.
        log_knorm = _log_kernel_norm(h, n_features, kernel_c)
        knorm = np.exp(log_knorm)

        # Loop over query points
        density = np.zeros(n_query)
        for i, point in enumerate(X):
            # Find distance between query point and each sample point.
            # Assumes Euclidean metric.
            dist = np.sqrt( ((self.data - point)**2.0).sum(axis=1) )

            # Find all the kernel values.
            kern = np.exp(compute_log_kernel(dist, h, kernel_c))

            # Sum up kernel values times weights to get the density.
            density[i] = (kern * self.weights).sum()
            
        log_density = np.log(density)

        # Normalize the results.
        log_density += log_knorm

        if return_log:
            return log_density
        else:
            return np.exp(log_density)


######################################################################
######################################################################
# Kernel functions
# Adapted to pure Python by JMG from Cython code in binary_tree.pxi.
#
# Note: Kernels assume dist is non-negative and h is positive
#       All kernel functions are normalized such that K(0, h) = 1.
#       The fully normalized kernel is:
#         K = exp[kernel_norm(h, d, kernel) + compute_kernel(dist, h, kernel)]
#       The code only works with non-negative kernels: i.e. K(d, h) >= 0
#       for all valid d and h.  Note that for precision, the log of both
#       the kernel and kernel norm is returned.
###cdef enum KernelType:
GAUSSIAN_KERNEL = 1
TOPHAT_KERNEL = 2
EPANECHNIKOV_KERNEL = 3
EXPONENTIAL_KERNEL = 4
LINEAR_KERNEL = 5
COSINE_KERNEL = 6

def kernel_code(kernel):
    """ Translate a string specification of a kernel to a kernel code.
    """
    if kernel == 'gaussian':
        kernel_c = GAUSSIAN_KERNEL
    elif kernel == 'tophat':
        kernel_c = TOPHAT_KERNEL
    elif kernel == 'epanechnikov':
        kernel_c = EPANECHNIKOV_KERNEL
    elif kernel == 'exponential':
        kernel_c = EXPONENTIAL_KERNEL
    elif kernel == 'linear':
        kernel_c = LINEAR_KERNEL
    elif kernel == 'cosine':
        kernel_c = COSINE_KERNEL
    else:
        raise ValueError("kernel = '%s' not recognized" % kernel)
    return kernel_c


log = np.log
cos = np.cos
sqrt = np.sqrt
from scipy.special import gammaln as lgamma

# some handy constants
INF = np.inf
NEG_INF = -np.inf
PI = np.pi
ROOT_2PI = sqrt(2 * PI)
LOG_PI = log(PI)
LOG_2PI = log(2 * PI)

###cdef inline DTYPE_t log_gaussian_kernel(DTYPE_t dist, DTYPE_t h):
def log_gaussian_kernel(dist, h):
    """log of the gaussian kernel for bandwidth h (unnormalized)"""
    return -0.5 * (dist * dist) / (h * h)


###cdef inline DTYPE_t log_tophat_kernel(DTYPE_t dist, DTYPE_t h):
def log_tophat_kernel(dist, h):
    """log of the tophat kernel for bandwidth h (unnormalized)"""
    result = np.zeros_like(dist)
    result[dist < h] = 0.0
    result[dist >= h] = NEG_INF
    return result
#     if dist < h:
#         return 0.0
#     else:
#         return NEG_INF


###cdef inline DTYPE_t log_epanechnikov_kernel(DTYPE_t dist, DTYPE_t h):
def log_epanechnikov_kernel(dist, h):
    """log of the epanechnikov kernel for bandwidth h (unnormalized)"""
    result = np.zeros_like(dist)
    result[dist < h] = log(1.0 - (dist * dist) / (h * h))
    result[dist >= h] = NEG_INF
    return result
#     if dist < h:
#         return log(1.0 - (dist * dist) / (h * h))
#     else:
#         return NEG_INF


###cdef inline DTYPE_t log_exponential_kernel(DTYPE_t dist, DTYPE_t h):
def log_exponential_kernel(dist, h):
    """log of the exponential kernel for bandwidth h (unnormalized)"""
    return -dist / h


###cdef inline DTYPE_t log_linear_kernel(DTYPE_t dist, DTYPE_t h):
def log_linear_kernel(dist, h):
    """log of the linear kernel for bandwidth h (unnormalized)"""
    result = np.zeros_like(dist)
    result[dist < h] = log(1 - dist / h)
    result[dist >= h] = NEG_INF
    return result
#     if dist < h:
#         return log(1 - dist / h)
#     else:
#         return NEG_INF


###cdef inline DTYPE_t log_cosine_kernel(DTYPE_t dist, DTYPE_t h):
def log_cosine_kernel(dist, h):
    """log of the cosine kernel for bandwidth h (unnormalized)"""
    result = np.zeros_like(dist)
    result[dist < h] = log(cos(0.5 * PI * dist / h))
    result[dist >= h] = NEG_INF
    return result
#     if dist < h:
#         return log(cos(0.5 * PI * dist / h))
#     else:
#         return NEG_INF


#cdef inline DTYPE_t compute_log_kernel(DTYPE_t dist, DTYPE_t h,
#                                       KernelType kernel):
def compute_log_kernel(dist, h, kernel):
    """Given a KernelType enumeration, compute the appropriate log-kernel"""
    if kernel == GAUSSIAN_KERNEL:
        return log_gaussian_kernel(dist, h)
    elif kernel == TOPHAT_KERNEL:
        return log_tophat_kernel(dist, h)
    elif kernel == EPANECHNIKOV_KERNEL:
        return log_epanechnikov_kernel(dist, h)
    elif kernel == EXPONENTIAL_KERNEL:
        return log_exponential_kernel(dist, h)
    elif kernel == LINEAR_KERNEL:
        return log_linear_kernel(dist, h)
    elif kernel == COSINE_KERNEL:
        return log_cosine_kernel(dist, h)


#------------------------------------------------------------
# Kernel norms are defined via the volume element V_n
# and surface element S_(n-1) of an n-sphere.
###cdef DTYPE_t logVn(ITYPE_t n):
def logVn(n):
    """V_n = pi^(n/2) / gamma(n/2 - 1)"""
    return 0.5 * n * LOG_PI - lgamma(0.5 * n + 1)


###cdef DTYPE_t logSn(ITYPE_t n):
def logSn(n):
    """V_(n+1) = int_0^1 S_n r^n dr"""
    return LOG_2PI + logVn(n - 1)


###cdef DTYPE_t _log_kernel_norm(DTYPE_t h, ITYPE_t d,
###                              KernelType kernel) except -1:
def _log_kernel_norm(h, d, kernel):
    """Given a KernelType enumeration, compute the kernel normalization.

    h is the bandwidth, d is the dimension.
    """
#     cdef DTYPE_t tmp, factor = 0
#     cdef ITYPE_t k
    tmp = 0
    factor = 0
    k = 0
    if kernel == GAUSSIAN_KERNEL:
        factor = 0.5 * d * LOG_2PI
    elif kernel == TOPHAT_KERNEL:
        factor = logVn(d)
    elif kernel == EPANECHNIKOV_KERNEL:
        factor = logVn(d) + log(2. / (d + 2.))
    elif kernel == EXPONENTIAL_KERNEL:
        factor = logSn(d - 1) + lgamma(d)
    elif kernel == LINEAR_KERNEL:
        factor = logVn(d) - log(d + 1.)
    elif kernel == COSINE_KERNEL:
        # this is derived from a chain rule integration
        factor = 0
        tmp = 2. / PI
        for k in range(1, d + 1, 2):
            factor += tmp
            tmp *= -(d - k) * (d - k - 1) * (2. / PI) ** 2
        factor = log(factor) + logSn(d - 1)
    else:
        raise ValueError("Kernel code not recognized")
    return -factor - d * log(h)


def kernel_norm(h, d, kernel, return_log=False):
    """Given a string specification of a kernel, compute the normalization.

    Parameters
    ----------
    h : float
        the bandwidth of the kernel
    d : int
        the dimension of the space in which the kernel norm is computed
    kernel : string
        The kernel identifier.  Must be one of
        ['gaussian'|'tophat'|'epanechnikov'|
         'exponential'|'linear'|'cosine']
    return_log : boolean
        if True, return the log of the kernel norm.  Otherwise, return the
        kernel norm.
    Returns
    -------
    knorm or log_knorm : float
        the kernel norm or logarithm of the kernel norm.
    """
    if kernel == 'gaussian':
        result = _log_kernel_norm(h, d, GAUSSIAN_KERNEL)
    elif kernel == 'tophat':
        result = _log_kernel_norm(h, d, TOPHAT_KERNEL)
    elif kernel == 'epanechnikov':
        result = _log_kernel_norm(h, d, EPANECHNIKOV_KERNEL)
    elif kernel == 'exponential':
        result = _log_kernel_norm(h, d, EXPONENTIAL_KERNEL)
    elif kernel == 'linear':
        result = _log_kernel_norm(h, d, LINEAR_KERNEL)
    elif kernel == 'cosine':
        result = _log_kernel_norm(h, d, COSINE_KERNEL)
    else:
        raise ValueError('kernel not recognized')

    if return_log:
        return result
    else:
        return np.exp(result)

