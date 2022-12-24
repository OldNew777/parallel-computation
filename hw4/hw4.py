import numpy as np

def conjugate_gradient(A, b, x0, tol=1e-4):
    """Solves the linear system Ax = b using the conjugate gradient method.

    Parameters
    ----------
    A : ndarray
        The matrix A in the linear system Ax = b.
    b : ndarray
        The vector b in the linear system Ax = b.
    x0 : ndarray
        The initial guess for the solution x.
    tol : float, optional
        The tolerance for the stopping criterion.

    Returns
    -------
    x : ndarray
        The solution to the linear system Ax = b.
    """
    x = x0.copy()
    r = b - A @ x
    p = r.copy()
    rr = r @ r
    iter = 0
    while np.linalg.norm(r) > tol * np.linalg.norm(x):
        # print('=================== Iter {:04d} =================='.format(iter))
        # print('p', p)
        # print('rr', rr)
        Ap = A @ p
        # print('Ap', Ap)
        alpha = rr / (p @ Ap)
        # print('alpha', alpha)
        x += alpha * p
        # print('x', x)
        r -= alpha * Ap
        # print('r', r)
        rr_new = r @ r
        # print('rr_new', rr_new)
        beta = rr_new / rr
        # print('beta', beta)
        # print(f'p({p}) * beta({beta})', p * beta)
        p = r + beta * p
        rr = rr_new
        iter += 1
        break
    return x


def gradient_descent(A, b, x0, tol=1e-4):
    """Solves the linear system Ax = b using the gradient descent method.

    Parameters
    ----------
    A : ndarray
        The matrix A in the linear system Ax = b.
    b : ndarray
        The vector b in the linear system Ax = b.
    x0 : ndarray
        The initial guess for the solution x.
    tol : float, optional
        The tolerance for the stopping criterion.

    Returns
    -------
    x : ndarray
        The solution to the linear system Ax = b.
    """
    x = x0.copy()
    r = b - A @ x
    iter = 0
    while np.linalg.norm(r) > tol * np.linalg.norm(x):
        # print('=================== Iter {:04d} =================='.format(iter))
        # print('r', r)
        alpha = (r @ r) / (r @ A @ r)
        # print('alpha', alpha)
        x += alpha * r
        # print('x', x)
        r -= alpha * A @ r
        # print('r', r)
        iter += 1
    return x


if __name__ == '__main__':
    n = 100
    m = 10
    A = np.array(np.random.rand(n, m), dtype=np.float64)
    b = np.array(np.random.rand(n), dtype=np.float64)
    x0 = np.array(np.random.rand(m), dtype=np.float64)
    print(conjugate_gradient(A, b, x0))

    # A = np.array([[0., -0.6027769223047823], [0.16918873018451275, 0.]], dtype=np.float64)
    # b = np.array([0.2777729740695085, 0.014353244778358086], dtype=np.float64)
    # x0 = np.array([0.08483570248860439, -0.46082217780902046], dtype=np.float64) + 0.1
    # print(conjugate_gradient(A, b, x0))
    # print(gradient_descent(A, b, x0))
    
    # A = np.array([[0.4, 0.1], [0.1, 0.3]], dtype=np.float64)
    # b = np.array([0.1, 0.2], dtype=np.float64)
    # x0 = np.zeros(shape=2, dtype=np.float64)
    # print(conjugate_gradient(A, b, x0))
    # print(gradient_descent(A, b, x0))

    # A = np.array(
    #     [[10, 1, 2, 3, 4],
    #     [4, 9, -3, 1, -3],
    #     [2, -1, 7, 3, -5],
    #     [3, 2, 3, 12, -2],
    #     [4, -3, -5, -1, 15]],
    #     dtype=np.float64
    # )
    # b = np.array([12, -27, 14, 17, 12], dtype=np.float64)
    # x0 = np.array([1, 1, 1, 1, 1], dtype=np.float64)
    # print(conjugate_gradient(A, b, x0))
