# Based on https://github.com/scikit-learn/scikit-learn/blob/a3305e688e8af3fed08aaaf85f96921ea2f386ab/sklearn/linear_model/_cd_fast.pyx
# License: BSD 3 clause

from libc.math cimport fabs
cimport cython 
import cython
cimport numpy as cnp
import numpy as np
import warnings
from cython cimport floating

from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._cython_blas cimport (
    ColMajor,
    Trans,
    NoTrans,
    _axpy,
    _dot,
    _asum,
    _gemv,
    _nrm2,
    _copy,
    _scal,
)
from sklearn.utils._random cimport our_rand_r

ctypedef cnp.float64_t DOUBLE
ctypedef cnp.uint32_t UINT32_t

cnp.import_array()

cdef enum:
    RAND_R_MAX = 0x7FFFFFFF


cdef inline UINT32_t rand_int(UINT32_t end, UINT32_t* random_state) noexcept nogil:
    """Generate a random integer in [0; end)."""
    return our_rand_r(random_state) % end


cdef inline floating fmax(floating x, floating y) noexcept nogil:
    if x > y:
        return x
    return y


cdef inline floating fmin(floating x, floating y) noexcept nogil:
    if x > y:
        return y
    return x


cdef inline floating fsign(floating f) noexcept nogil:
    if f == 0.0:
        return 0.0
    elif f > 0.0:
        return 1.0
    else:
        return -1.0


cdef floating abs_max(int n, floating* a) noexcept nogil:
    """np.max(np.abs(a))"""
    cdef int i
    cdef floating m = fabs(a[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i])
        if d > m:
            m = d
    return m


cdef floating max(int n, floating* a) noexcept nogil:
    """np.max(a)"""
    cdef int i
    cdef floating m = a[0]
    cdef floating d
    for i in range(1, n):
        d = a[i]
        if d > m:
            m = d
    return m


cdef floating diff_abs_max(int n, floating* a, floating* b) noexcept nogil:
    """np.max(np.abs(a - b))"""
    cdef int i
    cdef floating m = fabs(a[0] - b[0])
    cdef floating d
    for i in range(1, n):
        d = fabs(a[i] - b[i])
        if d > m:
            m = d
    return m

@cython.boundscheck(False)
@cython.wraparound(False) 
def enet_coordinate_descent(
    cnp.ndarray[floating, ndim=1, mode='c'] w,
    floating alpha,
    floating beta,
    cnp.ndarray[floating, ndim=2, mode='fortran'] X,
    cnp.ndarray[floating, ndim=1, mode='c'] y,
    cnp.ndarray[floating, ndim=1, mode='c'] constraints_sign,
    cnp.ndarray[floating, ndim=1, mode='c'] constraints_bound_upper,
    cnp.ndarray[floating, ndim=1, mode='c'] constraints_bound_lower,
    object rng,
    unsigned int max_iter=1000,
    floating tol=1e-4,
    bint random=0,
):
    """Cython version of the coordinate descent algorithm
        for Elastic-Net regression

        We minimize

        (1/2) * norm(y - X w, 2)^2 + alpha norm(w, 1) + (beta/2) norm(w, 2)^2

    Returns
    -------
    w : ndarray of shape (n_features,)
        ElasticNet coefficients.
    gap : float
        Achieved dual gap.
    tol : float
        Equals input `tol` times `np.dot(y, y)`. The tolerance used for the dual gap.
    n_iter : int
        Number of coordinate descent iterations.
    """

    if floating is float:
        dtype = np.float32
    else:
        dtype = np.float64

    cdef unsigned int n_samples = X.shape[0]
    cdef unsigned int n_features = X.shape[1]

    cdef floating[::1] norm_cols_X = np.square(X).sum(axis=0)
    cdef floating[::1] R = np.empty(n_samples, dtype=dtype)
    cdef floating[::1] XtA = np.empty(n_features, dtype=dtype)

    cdef floating tmp
    cdef floating w_ii
    cdef floating d_w_max
    cdef floating w_max
    cdef floating d_w_ii
    cdef floating gap = tol + 1.0
    cdef floating d_w_tol = tol
    cdef floating dual_norm_XtA
    cdef floating R_norm2
    cdef floating w_norm2
    cdef floating l1_norm
    cdef floating const
    cdef floating A_norm2
    cdef unsigned int ii
    cdef unsigned int i
    cdef unsigned int n_iter = 0
    cdef unsigned int f_iter
    cdef UINT32_t rand_r_state_seed = rng.randint(0, RAND_R_MAX)
    cdef UINT32_t* rand_r_state = &rand_r_state_seed

    if alpha == 0.0 and beta == 0.0:
        warnings.warn("Coordinate descent with no regularization may lead to "
                      "unexpected results and is discouraged.")

    with nogil:
        # R = y - np.dot(X, w)
        _copy(n_samples, &y[0], 1, &R[0], 1)
        _gemv(ColMajor, NoTrans, n_samples, n_features, -1.0, &X[0, 0], n_samples, &w[0], 1, 1.0, &R[0], 1)

        # tol *= np.dot(y, y)
        tol *= _dot(n_samples, &y[0], 1, &y[0], 1)

        for n_iter in range(max_iter):
            w_max = 0.0
            d_w_max = 0.0
            for f_iter in range(n_features):
                if random:
                    ii = rand_int(n_features, rand_r_state)
                else:
                    ii = f_iter

                if norm_cols_X[ii] == 0.0:
                    continue

                w_ii = w[ii]

                if w_ii != 0.0:
                    # R += w_ii * X[:,ii]
                    _axpy(n_samples, w_ii, &X[0, ii], 1, &R[0], 1)

                # tmp = (X[:,ii]*R).sum()
                tmp = _dot(n_samples, &X[0, ii], 1, &R[0], 1)

                if constraints_sign[ii] * tmp < 0.0:
                    w[ii] = 0.0
                else:
                    w[ii] = (fsign(tmp) * fmax(fabs(tmp) - alpha, 0.0) / (norm_cols_X[ii] + beta))
                    w[ii] = fmax(constraints_bound_lower[ii], fmin(constraints_bound_upper[ii], w[ii]))

                if w[ii] != 0.0:
                    # R -=  w[ii] * X[:,ii] # Update residual
                    _axpy(n_samples, -w[ii], &X[0, ii], 1, &R[0], 1)

                d_w_ii = fabs(w[ii] - w_ii)
                d_w_max = fmax(d_w_max, d_w_ii)

                w_max = fmax(w_max, fabs(w[ii]))

            if (
                w_max == 0.0
                or d_w_max / w_max < d_w_tol
                or n_iter == max_iter - 1
            ):
                # XtA = np.dot(X.T, R) - beta * w
                _copy(n_features, &w[0], 1, &XtA[0], 1)
                _gemv(ColMajor, Trans, n_samples, n_features, 1.0, &X[0, 0], n_samples, &R[0], 1, -beta, &XtA[0], 1)

                dual_norm_XtA = abs_max(n_features, &XtA[0])

                # R_norm2 = np.dot(R, R)
                R_norm2 = _dot(n_samples, &R[0], 1, &R[0], 1)

                # w_norm2 = np.dot(w, w)
                w_norm2 = _dot(n_features, &w[0], 1, &w[0], 1)

                if (dual_norm_XtA > alpha):
                    const = alpha / dual_norm_XtA
                    A_norm2 = R_norm2 * (const ** 2)
                    gap = 0.5 * (R_norm2 + A_norm2)
                else:
                    const = 1.0
                    gap = R_norm2

                l1_norm = _asum(n_features, &w[0], 1)

                # np.dot(R.T, y)
                gap +=  alpha * l1_norm - const * _dot(n_samples, &R[0], 1, &y[0], 1) + 0.5 * beta * (1 + const ** 2) * w_norm2

                if gap < tol:
                    break

        else:
            with gil:
                message = (
                    "Objective did not converge. You might want to increase "
                    "the number of iterations, check the scale of the "
                    "features or consider increasing regularisation. "
                    f"Duality gap: {gap:.3e}, tolerance: {tol:.3e}"
                )
                if alpha < np.finfo(np.float64).eps:
                    message += (
                        " Linear regression models with null weight for the "
                        "l1 regularization term are more efficiently fitted "
                        "using one of the solvers implemented in "
                        "sklearn.linear_model.Ridge/RidgeCV instead."
                    )
                warnings.warn(message, ConvergenceWarning)

    return np.asarray(w), gap, tol, n_iter + 1
