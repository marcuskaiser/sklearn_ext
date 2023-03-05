import unittest

import numpy as np
from cd import enet_coordinate_descent
from sklearn.linear_model import _cd_fast
from sklearn.utils.validation import check_random_state


class TestClass(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2023)
        x = rng.normal(size=(1000, 4))
        y = rng.normal(size=1000) + 0.01 * x[:, 0] - 0.01 * x[:, 1] + 0.1 * x[:, 2] - 0.1 * x[:, 3]

        self.fit_kwargs = dict(
            alpha=0.01,
            beta=0.01,
            X=np.asfortranarray(x, dtype='float64'),
            y=y,
            max_iter=1000,
            tol=1e-4,
            rng=check_random_state(0),
            random=0,
        )

    def test_0(self):
        w = np.zeros(4)
        constraints_sign = np.array([0.0, 0.0, 0.0, 0.0])
        constraints_bound_upper = np.array([np.inf] * 4)
        constraints_bound_lower = np.array([-np.inf] * 4)

        w, gap, tol, n_iter = enet_coordinate_descent(
            w=w,
            constraints_sign=constraints_sign,
            constraints_bound_upper=constraints_bound_upper,
            constraints_bound_lower=constraints_bound_lower,
            **self.fit_kwargs,
        )
        w_expected = np.array([0.02294786, -0.00508695, 0.10705516, -0.11733361])
        error = np.max(np.abs(w - w_expected))
        self.assertLessEqual(error, 1e-8)

        w2 = np.zeros(4)
        _cd_fast.enet_coordinate_descent(
            w=w2,
            **self.fit_kwargs,
        )
        error = np.max(np.abs(w - w2))
        self.assertLessEqual(error, 1e-8)

        from sklearn.linear_model import ElasticNet

        reg = ElasticNet(
            alpha=0.02 / self.fit_kwargs['X'].shape[0],
            l1_ratio=0.5 / self.fit_kwargs['X'].shape[0],
            fit_intercept=False,
            random_state=0,
            tol=1e-4,
        )
        reg.fit(
            self.fit_kwargs['X'],
            self.fit_kwargs['y'],
        )
        print(reg.coef_)
        error = np.max(np.abs(w - reg.coef_))
        self.assertLessEqual(error, 1e-8)

    def test_1(self):
        w = np.zeros(4)
        constraints_sign = np.array([0.0, 0.0, 0.0, 0.0])
        constraints_bound_upper = np.array([0.1, 0.1, 0.1, 0.1])
        constraints_bound_lower = np.array([-0.1, -0.1, -0.1, -0.1])

        w, gap, tol, n_iter = enet_coordinate_descent(
            w=w,
            constraints_sign=constraints_sign,
            constraints_bound_upper=constraints_bound_upper,
            constraints_bound_lower=constraints_bound_lower,
            **self.fit_kwargs,
        )

        self.assertLessEqual(np.max(np.abs(w)), 0.1)

        w_expected = np.array([0.0228036, -0.00439183, 0.1, -0.1])
        error = np.max(np.abs(w - w_expected))
        self.assertLessEqual(error, 1e-8)

    def test_2(self):
        w = np.zeros(4)
        constraints_sign = np.array([1.0, 1.0, 1.0, 1.0])
        constraints_bound_upper = np.array([np.inf] * 4)
        constraints_bound_lower = np.array([-np.inf] * 4)

        w, gap, tol, n_iter = enet_coordinate_descent(
            w=w,
            constraints_sign=constraints_sign,
            constraints_bound_upper=constraints_bound_upper,
            constraints_bound_lower=constraints_bound_lower,
            **self.fit_kwargs,
        )

        self.assertGreaterEqual(np.min(w), 0.0)

    def test_3(self):
        w = np.zeros(4)
        constraints_sign = np.array([-1.0, -1.0, 1.0, 1.0])
        constraints_bound_upper = np.array([np.inf] * 4)
        constraints_bound_lower = np.array([-np.inf] * 4)

        w, gap, tol, n_iter = enet_coordinate_descent(
            w=w,
            constraints_sign=constraints_sign,
            constraints_bound_upper=constraints_bound_upper,
            constraints_bound_lower=constraints_bound_lower,
            **self.fit_kwargs,
        )

        self.assertGreaterEqual(np.min(w * constraints_sign), 0.0)

    def test_4(self):
        w = np.zeros(4)
        constraints_sign = np.array([1.0, 1.0, -1.0, -1.0])
        constraints_bound_upper = np.array([np.inf] * 4)
        constraints_bound_lower = np.array([-np.inf] * 4)

        w, gap, tol, n_iter = enet_coordinate_descent(
            w=w,
            constraints_sign=constraints_sign,
            constraints_bound_upper=constraints_bound_upper,
            constraints_bound_lower=constraints_bound_lower,
            **self.fit_kwargs,
        )
        self.assertGreaterEqual(np.min(w * constraints_sign), 0.0)


if __name__ == '__main__':
    unittest.main()
