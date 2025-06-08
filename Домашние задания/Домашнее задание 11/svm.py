import numpy as np
from cvxopt import matrix
from cvxopt.solvers import qp
from cvxopt import solvers
from sklearn.svm import LinearSVC, SVC
from scipy.spatial.distance import cdist
import time as timer


class SVM:
    def __init__(self, C=1.0, method='primal', kernel=None, gamma=None):
        self._method_set = {'primal', 'dual', 'subgradient',
                            'stoch_subgradient', 'liblinear', 'libsvm'}
        self._kernel_set = {'linear', 'rbf', None}
        self._stop_criterion_set = {'objective', 'argument'}
        self._EPS = 0.000000001
        if (C < 0.0) or np.isclose(C, 0.0, rtol=0.0, atol=self._EPS) or not (method in self._method_set) or\
                ((method in {'libsvm', 'dual'}) and (not (kernel in {'linear', 'rbf'}) or
                                                     ((kernel == 'rbf') and ((gamma < 0) or np.isclose(gamma, 0.0, rtol=0.0, atol=self._EPS))))):
            raise ValueError('Some arguments are incorrect')
        self._C = float(C)
        self._method = method
        if method in {'libsvm', 'dual'}:
            self._kernel = kernel
            if kernel == 'rbf':
                self._gamma = float(gamma)

    @property
    def C(self):
        return self._C

    @property
    def method(self):
        return self._method

    @property
    def kernel(self):
        return self._kernel

    @property
    def gamma(self):
        return self._gamma

    def compute_primal_objective(self, X, y):
        if not (self._method in {'primal', 'subgradient', 'stoch_subgradient', 'liblinear'}) or not self._fitted:
            raise AttributeError(
                'Problem is not primal or classifier is not fitted')
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Arrays are not aligned by first component of shape')
        return np.sum(np.clip(1 - y * (np.dot(X, self._w) - self._intercept), 0, np.inf)) +\
            np.linalg.norm(self._w)**2 / (2.0 * self._C)

    def compute_dual_objective(self, X, y):
        if not (self._method in {'dual', 'libsvm'}) or not self._fitted:
            raise AttributeError(
                'Problem is not dual or classifier is not fitted')
        if (X.shape[0] != y.shape[0]) or (y.shape[0] != self._A.shape[0]):
            raise ValueError(
                'Arrays are not aligned by first component of shape')
        return self._A.sum() - 0.5 * (np.dot(y, y.T) * self._kernel_function(X, X) * np.dot(self._A, self._A.T)).sum()

    def fit(self, X, y, tol=0.0001, max_iter=1000, verbose=False, stop_criterion='objective', batch_size=20, lamb=0.05,
            alpha=0.1, beta=0.6):
        if (X.shape[0] != y.shape[0]) or (tol < 0.0) or np.isclose(tol, 0.0, rtol=0.0, atol=self._EPS*0.1) or \
                (max_iter < 1) or ((self._method in {'subgradient', 'stoch_subgradient'}) and
                                   (not (stop_criterion in self._stop_criterion_set) or
                                    ((self._method == 'stoch_subgradient') and ((batch_size < 1) or (batch_size > X.shape[0]) or
                                                                                ((stop_criterion == 'objective') and ((lamb < 0.0) or
                                                                                                                      np.isclose(lamb, 0.0, rtol=0.0, atol=self._EPS))))) or (alpha < 0.0) or
                                    np.isclose(alpha, 0.0, rtol=0.0, atol=self._EPS) or
                                    ((beta < 0.5) and not np.isclose(beta, 0.0, rtol=0.0, atol=self._EPS)) or
                                    np.isclose(beta, 0.5, rtol=0.0, atol=self._EPS))):
            raise ValueError('Some arguments are incorrect')
        self._fitted = True
        self._EPS = float(tol) * 0.1
        result = {}
        if self._method == 'primal':
            status = 0
            solvers.options['show_progress'] = verbose
            solvers.options['maxiters'] = max_iter
            solvers.options['abstol'] = tol
            solvers.options['reltol'] = tol
            solvers.options['feastol'] = tol
            P = np.zeros((1 + X.shape[1] + X.shape[0],
                         1 + X.shape[1] + X.shape[0]))
            idxs = np.arange(1, X.shape[1] + 1)
            P[idxs, idxs] = 1.0 / self._C
            P = matrix(P, tc='d')
            q = matrix(
                np.hstack((np.zeros(1 + X.shape[1]), np.ones(X.shape[0]))), tc='d')
            G = np.zeros((2 * X.shape[0], 1 + X.shape[1] + X.shape[0]))
            G[:X.shape[0], 0] = y.T[0]
            G[:X.shape[0], 1:X.shape[1] + 1] = -y * X
            idxs = np.arange(X.shape[0])
            G[idxs, 1 + X.shape[1] + idxs] = -1
            G[X.shape[0] + idxs, 1 + X.shape[1] + idxs] = -1
            G = matrix(G, tc='d')
            h = matrix(
                np.hstack((-np.ones(X.shape[0]), np.zeros(X.shape[0]))), tc='d')
            time = timer.time()
            solution = qp(P, q, G, h)
            time = timer.time() - time
            self._w = np.array(solution['x'])[1:X.shape[1] + 1, :]
            self._intercept = np.array(solution['x'][0, :])[0, 0]
            if solution['status'] != 'optimal':
                status = 1
            result = {'status': status, 'time': time}
        if self._method == 'dual':
            status = 0
            solvers.options['show_progress'] = verbose
            solvers.options['maxiters'] = max_iter
            solvers.options['abstol'] = tol
            solvers.options['reltol'] = tol
            solvers.options['feastol'] = tol
            P = matrix(self._kernel_function(X, X) * np.dot(y, y.T), tc='d')
            q = matrix(-np.ones(X.shape[0]), tc='d')
            G = matrix(np.vstack(
                (np.diag(-np.ones(X.shape[0])), np.diag(np.ones(X.shape[0])))), tc='d')
            h = matrix(
                np.hstack((np.zeros(X.shape[0]), self._C * np.ones(X.shape[0]))), tc='d')
            A = matrix(y, tc='d').T
            b = matrix(0, tc='d')
            time = timer.time()
            solution = qp(P, q, G, h, A, b)
            time = timer.time() - time
            self._A = np.array(solution['x'])
            self._A[self._A < self._EPS] = 0.0
            self._X = X.copy()
            self._y = y.copy()
            if solution['status'] != 'optimal':
                status = 1
            result = {'status': status, 'time': time}
        if self._method == 'liblinear':
            self._clf = LinearSVC(
                C=self._C, max_iter=max_iter, tol=tol, verbose=verbose, dual=False)
            time = timer.time()
            self._clf.fit(X, y.T[0])
            time = timer.time() - time
            self._intercept = -self._clf.intercept_[0]
            self._w = self._clf.coef_.T
            result = {'time': time}
        if self._method == 'libsvm':
            gamma = 1.0 / X.shape[0]
            if self.kernel == 'rbf':
                gamma = self._gamma
            self._clf = SVC(C=self._C, max_iter=max_iter, tol=tol,
                            verbose=verbose, kernel=self._kernel, gamma=gamma)
            time = timer.time()
            self._clf.fit(X, y.T[0])
            time = timer.time() - time
            self._A = np.zeros(X.shape[0])
            self._A[self._clf.support_] = self._clf.dual_coef_
            self._A = np.abs(self._A[:, np.newaxis])
            result = {'time': time}
        if self._method == 'subgradient':
            status = 0
            objective_curve = []
            time = timer.time()
            self._w = np.random.randn(X.shape[1], 1)
            self._intercept = np.random.randn()
            crit_value = 1.0
            prim_obj_val = np.clip(
                1 - y * (np.dot(X, self._w) - self._intercept), 0, np.inf)[:, 0]
            cur_objective_value = np.sum(
                prim_obj_val) + np.linalg.norm(self._w)**2 / (2.0 * self._C)
            if stop_criterion == 'objective':
                crit_value = cur_objective_value
            elif stop_criterion == 'argument':
                crit_value = np.vstack((self._w, self._intercept))
            prev_crit_value = 0.0
            d_crit_value = 1
            k = 1
            objective_curve.append(cur_objective_value)
            while (np.linalg.norm(d_crit_value) > tol) or\
                    np.isclose(np.linalg.norm(d_crit_value), tol, rtol=0.0, atol=self._EPS):
                prev_crit_value = crit_value
                gradient_w = self._w / self._C
                mask = np.where(prim_obj_val > 0.0)[0]
                gradient_w += (-y * X)[mask, :].sum(axis=0)[:, np.newaxis]
                gradient_intercept = y[mask].sum()
                step = float(alpha) / k**beta
                self._w -= step * gradient_w
                self._intercept -= step * gradient_intercept
                prim_obj_val = np.clip(
                    1 - y * (np.dot(X, self._w) - self._intercept), 0, np.inf)[:, 0]
                cur_objective_value = np.sum(
                    prim_obj_val) + np.linalg.norm(self._w)**2 / (2.0 * self._C)
                objective_curve.append(cur_objective_value)
                if stop_criterion == 'objective':
                    crit_value = cur_objective_value
                elif stop_criterion == 'argument':
                    crit_value = np.vstack((self._w, self._intercept))
                d_crit_value = crit_value - prev_crit_value
                if verbose:
                    print('iteration number {}, {} criterion value = {}'.format(k, stop_criterion,
                                                                                np.linalg.norm(d_crit_value)))
                if k == max_iter:
                    status = 1
                    break
                k += 1
            result = {'status': status, 'objective_curve': objective_curve,
                      'time': timer.time() - time}
        if self._method == 'stoch_subgradient':
            status = 0
            objective_curve = []
            time = timer.time()
            self._w = np.random.randn(X.shape[1], 1)
            self._intercept = np.random.randn()
            crit_value = 1.0
            prim_obj_val = np.clip(
                1 - y * (np.dot(X, self._w) - self._intercept), 0, np.inf)[:, 0]
            cur_objective_value = (np.sum(
                prim_obj_val) + np.linalg.norm(self._w)**2 / (2.0 * self._C)) / X.shape[0]
            if stop_criterion == 'objective':
                crit_value = cur_objective_value
            elif stop_criterion == 'argument':
                crit_value = np.vstack((self._w, self._intercept))
            prev_crit_value = 0.0
            d_crit_value = 1
            k = 1
            objective_curve.append(cur_objective_value)
            batch_idxs = np.random.choice(
                X.shape[0], batch_size, replace=False)
            prim_obj_val = np.clip(1 - y[batch_idxs] * (np.dot(X[batch_idxs, :], self._w) - self._intercept),
                                   0, np.inf)[:, 0]
            while (np.linalg.norm(d_crit_value) > tol) or\
                    np.isclose(np.linalg.norm(d_crit_value), tol, rtol=0.0, atol=self._EPS):
                prev_crit_value = crit_value
                gradient_w = self._w / self._C
                mask = np.where(prim_obj_val > 0.0)[0]
                gradient_w += (-y[batch_idxs] * X[batch_idxs, :]
                               )[mask, :].sum(axis=0)[:, np.newaxis]
                gradient_intercept = y[batch_idxs[mask]].sum()
                step = float(alpha) / k ** beta
                self._w -= step * gradient_w
                self._intercept -= step * gradient_intercept
                batch_idxs = np.random.choice(
                    X.shape[0], batch_size, replace=False)
                prim_obj_val = np.clip(1 - y[batch_idxs] * (np.dot(X[batch_idxs, :], self._w) - self._intercept),
                                       0, np.inf)[:, 0]
                cur_objective_value = (np.sum(
                    prim_obj_val) + np.linalg.norm(self._w)**2 / (2.0 * self._C)) / batch_size
                objective_curve.append(cur_objective_value)
                if stop_criterion == 'objective':
                    crit_value = (1.0 - lamb) * crit_value +\
                        lamb * cur_objective_value
                elif stop_criterion == 'argument':
                    crit_value = np.vstack((self._w, self._intercept))
                d_crit_value = crit_value - prev_crit_value
                if verbose:
                    print('iteration number {}, {} criterion value = {}'.format(k, stop_criterion,
                                                                                np.linalg.norm(d_crit_value)))
                if k == max_iter:
                    status = 1
                    break
                k += 1
            result = {'status': status, 'objective_curve': objective_curve,
                      'time': timer.time() - time}
        return result

    @property
    def w(self):
        return self._w

    @property
    def intercept(self):
        return self._intercept

    @property
    def A(self):
        return self._A

    def predict(self, X_test, return_classes=False):
        if not self._fitted:
            raise AttributeError('Classifier is not fitted')
        if self._method in {'primal', 'subgradient', 'stoch_subgradient'}:
            if return_classes:
                result = np.sign(np.dot(X_test, self._w) - self._intercept)
                result[result == 0] = 1
                return result
            else:
                return (np.dot(X_test, self._w) - self._intercept)
        elif self._method == 'dual':
            kernel_matrix = self._kernel_function(
                self._X, X_test) * self._y * self._A
            kernel_matrix = kernel_matrix.sum(axis=0)
            idxs = np.where((self._A > self._EPS) & (self._A < self._C))[0]
            if idxs.any():
                interception = self._kernel_function(
                    self._X, self._X[idxs, :]) * self._y * self._A
                kernel_matrix -= (interception.sum() -
                                  self._y[idxs].sum()) / idxs.size
            if return_classes:
                kernel_matrix = np.sign(kernel_matrix)
                kernel_matrix[kernel_matrix == 0] = 1
                return kernel_matrix[:, np.newaxis]
            return kernel_matrix[:, np.newaxis]
        else:
            if return_classes:
                return self._clf.predict(X_test)[:, np.newaxis]
            else:
                return self._clf.decision_function(X_test)[:, np.newaxis]

    def compute_support_vectors(self, X):
        if not (self._method in {'dual', 'libsvm'}) or not self._fitted or (X.shape[0] != self._A.shape[0]):
            raise AttributeError(
                'Problem is not dual or classifier is not fitted')
        if self._method == 'dual':
            mask = self._A > self._EPS
            return X[mask[:, 0], :].copy()
        else:
            return self._clf.support_vectors_.copy()

    def compute_w(self, X, y):
        if not (self._method in {'dual', 'libsvm'}) or \
                (self._kernel != 'linear') or not self._fitted:
            raise AttributeError(
                'Problem is not dual or kernel is not linear or classifier is not fitted')
        if (X.shape[0] != y.shape[0]) or (y.shape[0] != self._A.shape[0]):
            raise ValueError(
                'Arrays are not aligned by first component of shape')
        if self._method == 'dual':
            result_w = self._A * y
            result_w = np.einsum('ij->j', result_w * X)
            idxs = np.where((self._A > self._EPS) & (self._A < self._C))[0]
            if idxs.any():
                self._intercept = (
                    np.dot(result_w, X[idxs, :].T).sum() - y[idxs].sum()) / idxs.size
            else:
                self._intercept = 0.0
            return result_w[:, np.newaxis]
        else:
            self._intercept = -self._clf.intercept_[0]
            return self._clf.coef_.T.copy()

    def _kernel_function(self, x, y):
        if not (self._method in {'dual', 'libsvm'}):
            raise AttributeError('Problem is not dual')
        if self._kernel == 'linear':
            return np.dot(x, y.T)
        elif self._kernel == 'rbf':
            return np.exp(-self._gamma * cdist(x, y, 'euclidean')**2)
        else:
            raise AttributeError('Unknown type of kernel')

    @property
    def fitted(self):
        return self._fitted

    @property
    def EPS(self):
        return self._EPS


def visualize(X, y, alg_svm, show_vectors=False, title='Decision regions', x1='x1', x2='x2', return_classes=True):
    if (X.shape[0] != y.shape[0]) or (show_vectors and not (alg_svm.method in {'dual', 'libsvm'})) or\
            (X.shape[1] != 2) or not alg_svm.fitted:
        raise ValueError('Some arguments are incorrect')
    # visualize feature space in 2d inspired by: http://scikit-learn.org/stable/auto_examples/ensemble/
    # plot_voting_decision_regions.html#sphx-glr-auto-examples-ensemble-plot-voting-decision-regions-py
    import matplotlib.pyplot as plt
    from matplotlib import cm
    plt.close()
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    number_of_dots = 1000
    xx1, xx2 = np.meshgrid(np.linspace(
        x1_min, x1_max, number_of_dots), np.linspace(x2_min, x2_max, number_of_dots))
    f, axarr = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(10, 8))
    Z = alg_svm.predict(np.c_[xx1.ravel(), xx2.ravel()],
                        return_classes=return_classes).reshape(xx1.shape)
    axarr.contourf(xx1, xx2, Z, alpha=0.5, cmap=cm.seismic)
    axarr.contour(xx1, xx2, Z, colors=['k', 'k', 'k'], linestyles=[
                  '--', '-', '--'], levels=[-1.0, 0, 1.0])
    axarr.set_title(title)
    if show_vectors:
        SV = alg_svm.compute_support_vectors(X)
        axarr.scatter(SV[:, 0], SV[:, 1], s=60, facecolors='none')
    axarr.scatter(X[:, 0], X[:, 1], c=y.T[0], alpha=0.8)
    plt.xlabel(x1, fontsize=16)
    plt.ylabel(x2, fontsize=16)
    plt.show()
