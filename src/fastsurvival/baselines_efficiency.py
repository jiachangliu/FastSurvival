import functools
import gc
import math
import sys
from collections import deque
from itertools import chain
# from __future__ import print_function
from sys import getsizeof, stderr

import numpy as np
import psutil

try:
    from reprlib import repr
except ImportError:
    pass

from itertools import combinations

import scipy

from heapq import heappush, heappop

import math
import time



class coxModel_unnormalized_big_p:
    def __init__(self, X, y_time, y_event):
        self.sorted_indices = np.lexsort((-y_event, y_time))

        self.X = X[self.sorted_indices]
        self.y_time = y_time[self.sorted_indices]
        self.y_event = y_event[self.sorted_indices]

        self.X_squared = self.X ** 2

        self.n, self.p = self.X.shape

        y_event_time_list = [(self.y_event[i], self.y_time[i]) for i in range(self.n)]
        dtype = [('event', '?'), ('time', '<f8')]
        self.y_sksurv = np.array(y_event_time_list, dtype=dtype)

        self.y = np.concatenate([self.y_time[:, None], self.y_event[:, None]], axis=1)
        self.event_1_mask = self.y_event == 1

        self.same_time_first_event_indices = [0] if self.event_1_mask[0] else []
        self.same_time_same_event_counts = []

        count = 1 if self.event_1_mask[0] else 0
        for i in range(1, len(self.event_1_mask)) :
            if self.event_1_mask[i] and (self.y_time[i] != self.y_time[i-1]):
                self.same_time_first_event_indices.append(i)
                if count > 0:
                    self.same_time_same_event_counts.append(count)
                count = 1
            elif self.event_1_mask[i] and self.y_time[i] == self.y_time[i-1]:
                count += 1
            elif (not self.event_1_mask[i]) and self.event_1_mask[i-1]:
                self.same_time_same_event_counts.append(count)
                count = 0

        if count > 0:
            self.same_time_same_event_counts.append(count)

        self.same_time_first_event_indices = np.array(self.same_time_first_event_indices, dtype=int)
        self.same_time_same_event_counts = np.array(self.same_time_same_event_counts, dtype=int)

        if np.sum(self.same_time_same_event_counts) != np.sum(self.event_1_mask):
            raise ValueError(f"Something is wrong with the sum of same_time_same_event_counts (sum={np.sum(self.same_time_same_event_counts)}), which should be equal to the sum of event_1_mask (sum={np.sum(self.event_1_mask)})")

        if len(self.same_time_first_event_indices) != len(self.same_time_same_event_counts):
            raise ValueError(f"The length of same_time_first_event_indices {len(self.same_time_first_event_indices)} and same_time_same_event_counts {len(self.same_time_same_event_counts)} should be the same")


        self.w = np.zeros(self.p)
        self.lambda2 = 0.0
        self.twolambda2 = 2 * self.lambda2
        self.total_child_added = 0

        self.samplewise_diagHessianEntryWiseUpperBound = np.zeros((self.n,))

        self.samplewise_diagHessianEntryWiseUpperBound[self.same_time_first_event_indices] = self.same_time_same_event_counts * 0.25
        self.samplewise_diagHessianEntryWiseUpperBound = np.cumsum(self.samplewise_diagHessianEntryWiseUpperBound)

        self.featurewise_diagHessianUpperBound = np.zeros((self.p,))
        cummin_X = np.minimum.accumulate(self.X[::-1], axis=0)[::-1]
        cummax_X = np.maximum.accumulate(self.X[::-1], axis=0)[::-1]

        self.featurewise_diagHessianUpperBound = 0.25 * np.sum(self.same_time_same_event_counts.reshape(-1, 1) * (cummax_X[self.same_time_first_event_indices] - cummin_X[self.same_time_first_event_indices])** 2, axis=0)

        self.featurewise_diagCubicUpperBound = 1/(6 * math.sqrt(3)) * np.sum(self.same_time_same_event_counts.reshape(-1, 1) * (cummax_X[self.same_time_first_event_indices] - cummin_X[self.same_time_first_event_indices])** 3, axis=0)

        self.B = self.y_time.reshape(-1, 1) <= self.y_time.reshape(1, -1)

    def compute_cox_loss(self, ExpXw, w):
        first_term = np.sum(np.log(ExpXw)[self.event_1_mask])
        reverse_sum_ExpyXw = np.cumsum(ExpXw[::-1])[::-1]
        second_term = np.sum(np.log(reverse_sum_ExpyXw[self.same_time_first_event_indices]) * self.same_time_same_event_counts)

        return -first_term + second_term

    def compute_samplewise_gradient_and_hessian(self, ExpXw):
        reverse_sum_ExpXw = np.cumsum(ExpXw[::-1])[::-1]
        common_term = np.zeros((self.n,))
        common_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices]
        common_term = np.cumsum(common_term)
        common_term *= ExpXw

        samplewise_gradient = common_term - self.event_1_mask

        # second_term = np.zeros((self.n,))
        # second_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices] ** 2
        # second_term = np.cumsum(second_term)
        # second_term *= ExpXw ** 2
        # samplewise_diagHessian = second_term - common_term

        # samplewise_diagHessian = common_term
        samplewise_diagHessian = self.samplewise_diagHessianEntryWiseUpperBound

        return samplewise_gradient, samplewise_diagHessian

class coxModelElasticNet_newton_base(coxModel_unnormalized_big_p):
    def __init__(
        self,
        X,
        y_time,
        y_event,
        verbose=False,
        lambda1=0.0,
        lambda2=1e-3,
    ):
        super().__init__(X=X, y_time=y_time, y_event=y_event)

        self.w = np.zeros((self.p,))
        self.ExpXw = np.exp(np.zeros((self.n,)))

        self.verbose = verbose
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.twolambda2 = 2 * lambda2

    def reset_coef(self, w):
        self.w[:] = w[:]
        self.ExpXw = np.exp(self.X.dot(self.w))

    def compute_elastic_net_penalty(self, w):
        return self.lambda1 * np.sum(np.abs(w)) + self.lambda2 * np.sum(w ** 2)

    def compute_total_loss(self):
        return self.compute_cox_loss(self.ExpXw, self.w) + self.compute_elastic_net_penalty(self.w)

    def get_samplewise_gradient(self):
        reverse_sum_ExpXw = np.cumsum(self.ExpXw[::-1])[::-1]
        common_term = np.zeros((self.n,))
        common_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices]
        common_term = np.cumsum(common_term)
        common_term *= self.ExpXw
        samplewise_gradient = common_term - self.event_1_mask

        # common_term_2 = np.diag(self.ExpXw) @ self.B.T @ (self.y_event / (self.B @ self.ExpXw))
        # print(np.sum(np.abs(common_term - common_term_2)))

        return samplewise_gradient

    def get_samplewise_hessian(self):
        pass

    def get_feature_gradient(self):
        return self.X.T.dot(self.get_samplewise_gradient())

    def get_feature_hessian(self):
        return self.X.T.dot(self.get_samplewise_hessian()).dot(self.X)

    def get_feature_gradient_j(self, j):
        return self.X[:, j].T.dot(self.get_samplewise_gradient())

    def get_feature_hessian_j(self, j):
        return self.X[:, j].T.dot(self.get_samplewise_hessian()).dot(self.X[:, j])

    def fit(self, max_iter=300, coordinate=-1):
        iters = [0]
        losses = [self.compute_total_loss()]
        elapsed_times = [0]
        start_time = time.time()

        print(f"i:{0}, loss:{losses[-1]}, elapsed_time:{elapsed_times[-1]}")
        for i in range(1, 1+max_iter):
            iters.append(i)

            if coordinate == -1:
                self.optimize_all_coordinates_per_iteration()
            else:
                self.optimize_coordinate_j_per_iteration(coordinate)

            loss = self.compute_total_loss()
            losses.append(loss)
            elapsed_time = time.time() - start_time
            elapsed_times.append(elapsed_time)

            print(f"i:{i}, loss:{loss}, elapsed_time:{elapsed_time}")

            if loss > 1e8:
                print(f"loss is too high: {loss}")
                break

        print(f"nonzero indices: {np.nonzero(self.w)}, num nonzero: {np.count_nonzero(self.w)}, p: {self.p}")
        return iters, losses, elapsed_times

    def optimize_coordinate_j_per_iteration(self, j):
        pass

    def optimize_all_coordinates_per_iteration(self):
        pass

def fill_matrix_optimized(diagonal_entries):
    '''
    given a vector with size n, corresponding to the diagonal of matrix A whose diagonal entries are nonzero and off-diagonal entries are zero, how to return matrix A will off-diagonal entries filled so that A_{ij}= A{ii} if i <= j and A_{ij} = A_{jj} otherwise.
    '''
    n = len(diagonal_entries)
    A = np.diag(diagonal_entries)
    row_indices, col_indices = np.indices((n, n))
    A[row_indices <= col_indices] = diagonal_entries[row_indices[row_indices <= col_indices]]
    A[row_indices > col_indices] = diagonal_entries[col_indices[row_indices > col_indices]]
    return A

class coxModelElasticNet_newton_exact(coxModelElasticNet_newton_base):
    def __init__(
        self,
        X,
        y_time,
        y_event,
        verbose=False,
        lambda1=0.0,
        lambda2=1e-3,
    ):
        if lambda1 != 0:
            raise ValueError("lambda1 should be 0 for newton_exact")
        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

    def get_samplewise_hessian(self):
        reverse_sum_ExpXw = np.cumsum(self.ExpXw[::-1])[::-1]
        common_term = np.zeros((self.n,))
        common_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices]
        common_term = np.cumsum(common_term)
        common_term *= self.ExpXw

        # time_start = time.time()

        second_term = np.zeros((self.n,))
        second_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices] ** 2
        second_term = np.cumsum(second_term)
        second_term = fill_matrix_optimized(second_term) * (self.ExpXw.reshape(-1, 1) * self.ExpXw.reshape(1, -1))

        # time_v1 = time.time() - time_start
        # BExpXw = self.B @ self.ExpXw
        # BdiagExpXw = self.B @ np.diag(self.ExpXw)
        # second_term_v2 = BdiagExpXw.T @ np.diag(self.y_event / BExpXw**2) @ BdiagExpXw
        # time_v2 = time.time() - time_start - time_v1
        # print(f"difference is {np.sum(np.abs(second_term - second_term_v2))}")
        # print(f"time_v1: {time_v1}, time_v2: {time_v2}")
        # sys.exit()

        samplewise_hessian = np.diag(common_term) - second_term


        return samplewise_hessian

    def optimize_all_coordinates_per_iteration(self):
        gradient = self.get_feature_gradient() + self.twolambda2 * self.w
        hessian = self.get_feature_hessian() + self.twolambda2 * np.eye(self.p)

        self.w -= np.linalg.solve(hessian, gradient)
        self.ExpXw = np.exp(self.X.dot(self.w))

    def optimize_coordinate_j_per_iteration(self, j):
        gradient_j = self.get_feature_gradient_j(j) + self.twolambda2 * self.w[j]
        hessian_j = self.get_feature_hessian_j(j) + self.twolambda2

        self.w[j] -= gradient_j / hessian_j
        self.ExpXw = np.exp(self.X.dot(self.w))

def soft_thresholding_operator(x, threshold):
    return  math.copysign(max(abs(x) - threshold, 0), x)

class coxModelElasticNet_newton_quasi(coxModelElasticNet_newton_base):
    def __init__(
        self,
        X,
        y_time,
        y_event,
        verbose=False,
        lambda1=0.0,
        lambda2=1e-3,
    ):
        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

        self.sum_X_squared_axis_0 = np.sum(self.X_squared, axis=0)
        self.dw_all_zeros = np.zeros((self.p,))

    def get_samplewise_hessian(self):
        reverse_sum_ExpXw = np.cumsum(self.ExpXw[::-1])[::-1]
        common_term = np.zeros((self.n,))
        common_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices]
        common_term = np.cumsum(common_term)
        common_term *= self.ExpXw

        second_term = np.zeros((self.n,))
        second_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices] ** 2
        second_term = np.cumsum(second_term)
        second_term *= self.ExpXw ** 2

        samplewise_hessian = np.diag(common_term - second_term)

        return samplewise_hessian

    def get_soft_thresholding_input_at_coordinate_j(self, j, dw, w, gradient, hessian):
        a = 0.5 * hessian[j, j]
        b = gradient[j] + hessian[:, j].dot(dw) - hessian[j, j] * dw[j]
        soft_threshold_x = w[j] - 0.5 * b / a
        soft_threshold_lambda = self.lambda1 / a

        return soft_threshold_x, soft_threshold_lambda

    def optimize_all_coordinates_per_iteration(self):
        gradient = self.get_feature_gradient() + self.twolambda2 * self.w
        hessian = self.get_feature_hessian() + self.twolambda2 * np.eye(self.p)

        dw = np.zeros((self.p,))
        for iter in range(1000):
            dw_prev = dw.copy()
            for j in range(self.p):
                soft_threshold_x, soft_threshold_lambda = self.get_soft_thresholding_input_at_coordinate_j(j, dw, self.w, gradient, hessian)
                dw[j] = soft_thresholding_operator(soft_threshold_x, soft_threshold_lambda) - self.w[j]
            if np.linalg.norm(dw - dw_prev) < 1e-6:
                break

        self.w += dw
        self.ExpXw = np.exp(self.X.dot(self.w))

    def optimize_coordinate_j_per_iteration(self, j):
        gradient_j = self.get_feature_gradient_j(j) + self.twolambda2 * self.w[j]
        hessian_j = self.get_feature_hessian_j(j) + self.twolambda2

        a, b = 0.5 * hessian_j, gradient_j
        soft_threshold_x, soft_threshold_lambda = self.w[j] - 0.5 * b / a, self.lambda1 / a
        dw_j = soft_thresholding_operator(soft_threshold_x, soft_threshold_lambda) - self.w[j]

        self.w[j] += dw_j
        self.ExpXw = np.exp(self.X.dot(self.w))

class coxModelElasticNet_newton_prox(coxModelElasticNet_newton_quasi):
    def __init__(
        self,
        X,
        y_time,
        y_event,
        verbose=False,
        lambda1=0.0,
        lambda2=1e-3,
    ):
        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

    def get_samplewise_hessian(self):
        reverse_sum_ExpXw = np.cumsum(self.ExpXw[::-1])[::-1]
        common_term = np.zeros((self.n,))
        common_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices]
        common_term = np.cumsum(common_term)
        common_term *= self.ExpXw

        samplewise_hessian = np.diag(common_term)

        return samplewise_hessian


class coxModelElasticNet_newton_quadratic_surrogate(coxModelElasticNet_newton_base):
    def __init__(
        self,
        X,
        y_time,
        y_event,
        verbose=False,
        lambda1=0.0,
        lambda2=1e-3,
    ):
        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

    def get_feature_gradient_at_coordinate_j(self, j):
        samplewise_gradient = self.get_samplewise_gradient()
        return self.X[:, j].dot(samplewise_gradient) + self.twolambda2 * self.w[j]

    def get_soft_thresholding_input_at_coordinate_j(self, j, dw, w_j, gradient_j, L_j):
        a = 0.5 * L_j
        b = gradient_j
        soft_threshold_x = w_j - 0.5 * b / a
        soft_threshold_lambda = self.lambda1 / a

        return soft_threshold_x, soft_threshold_lambda

    def optimize_coordinate_j_per_iteration(self, j):
        grad_j = self.get_feature_gradient_at_coordinate_j(j)
        L2_j = self.featurewise_diagHessianUpperBound[j] + self.twolambda2

        # dw_j = - grad_j / L_j
        # self.w[j] += dw_j
        # self.ExpXw *= np.exp(self.X[:, j].dot(dw_j))

        soft_threshold_x, soft_threshold_lambda = self.get_soft_thresholding_input_at_coordinate_j(j, 0, self.w[j], grad_j, L2_j)
        dw_j = soft_thresholding_operator(soft_threshold_x, soft_threshold_lambda) - self.w[j]
        self.w[j] += dw_j
        self.ExpXw *= np.exp(self.X[:, j].dot(dw_j))

    def optimize_all_coordinates_per_iteration(self):
        for j in range(self.p):
            self.optimize_coordinate_j_per_iteration(j)

class coxModelElasticNet_newton_quadratic_surrogate_accelerated(coxModelElasticNet_newton_quadratic_surrogate):
    def __init__(
            self,
            X,
            y_time,
            y_event,
            verbose=False,
            lambda1=0.0,
            lambda2=1e-3,
    ):
        if lambda1 != 0:
            raise ValueError("lambda1 should be 0 for newton_quadratic_surrogate_accelerated")

        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

        self.theta = 1

        self.u = self.w.copy()
        self.v = self.w.copy()

        self.prob = (self.featurewise_diagHessianUpperBound + self.twolambda2) / np.sum(self.featurewise_diagHessianUpperBound + self.twolambda2)
        self.prob = (1 / self.prob) / np.sum(1 / self.prob)

    def optimize_coordinate_j_per_iteration(self, j):
        dw_j = (1-self.theta) * self.u[j] + self.theta * self.v[j] - self.w[j]
        self.w[j] += dw_j
        self.ExpXw *= np.exp(self.X[:, j].dot(dw_j))

        # self.w = (1-self.theta) * self.u + self.theta * self.v
        # self.ExpXw = np.exp(self.X.dot(self.w))

        L2_j = self.featurewise_diagHessianUpperBound[j] + self.twolambda2
        grad_j = self.get_feature_gradient_at_coordinate_j(j)

        self.u[j] = self.w[j] - 1 / L2_j * grad_j
        self.v[j] = self.v[j] - 1 / (L2_j * self.p * self.theta) * grad_j

        self.theta = (-self.theta ** 2 + self.theta * math.sqrt(self.theta ** 2 + 4)) / 2

        # dw_j = (1-self.theta) * self.u[j] + self.theta * self.v[j] - self.w[j]
        # self.w[j] += dw_j
        # self.ExpXw *= np.exp(self.X[:, j].dot(dw_j))


    def optimize_all_coordinates_per_iteration(self):
        for j in range(self.p):
            self.optimize_coordinate_j_per_iteration(j)

class coxModelElasticNet_newton_quadratic_surrogate_accelerated_HaihaoLu(coxModelElasticNet_newton_quadratic_surrogate_accelerated):
    def __init__(
            self,
            X,
            y_time,
            y_event,
            verbose=False,
            lambda1=0.0,
            lambda2=1e-3,
    ):
        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

    def optimize_coordinate_j_per_iteration(self, j):

        old_u = self.u.copy()
        self.u = (1 - self.theta) * self.w + self.theta * self.v
        diff_u = self.u - old_u
        print(f"diff_u nonzero indices: {np.nonzero(diff_u)}")

        diff_w = self.u - self.w
        print(f"diff_w nonzero indices: {np.nonzero(diff_w)}")

        self.w[:] = self.u[:]
        self.ExpXw = np.exp(self.X @ self.w)

        L2_j = self.featurewise_diagHessianUpperBound[j] + self.twolambda2
        grad_j = self.get_feature_gradient_at_coordinate_j(j)

        self.w[j] = self.w[j] - 1 / L2_j * grad_j

        old_v = self.v.copy()
        self.v[j] = self.v[j] - 1 / (L2_j * self.p * self.theta) * grad_j
        diff_v = self.v - old_v
        print(f"diff_v nonzero indices: {np.nonzero(diff_v)}")
        print()

        self.ExpXw /= np.exp(self.X[:, j] / L2_j * grad_j)

        self.theta = (-self.theta ** 2 + self.theta * math.sqrt(self.theta ** 2 + 4)) / 2

class coxModelElasticNet_newton_cubic_surrogate(coxModelElasticNet_newton_quadratic_surrogate):
    def __init__(
        self,
        X,
        y_time,
        y_event,
        verbose=False,
        lambda1=0.0,
        lambda2=1e-3,
    ):
        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

    def get_samplewise_hessian(self):
        print("get samplewise hessian")
        reverse_sum_ExpXw = np.cumsum(self.ExpXw[::-1])[::-1]
        common_term = np.zeros((self.n,))
        common_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices]
        common_term = np.cumsum(common_term)
        common_term *= self.ExpXw

        second_term = np.zeros((self.n,))
        second_term[self.same_time_first_event_indices] = self.same_time_same_event_counts / reverse_sum_ExpXw[self.same_time_first_event_indices] ** 2
        second_term = np.cumsum(second_term)
        second_term = fill_matrix_optimized(second_term) * (self.ExpXw.reshape(-1, 1) * self.ExpXw.reshape(1, -1))

        samplewise_hessian = np.diag(common_term) - second_term

        return samplewise_hessian

    def get_feature_hessian_at_coordinate_j_slow(self, j):
        samplewise_hessian = self.get_samplewise_hessian()
        return self.X[:, j].dot(samplewise_hessian.dot(self.X[:, j])) + self.twolambda2

    def get_feature_hessian_at_coordinate_j(self, j):
        reverse_sum_ExpXw = np.cumsum(self.ExpXw[::-1])[::-1]
        reverse_sum_ExpXw_X = np.cumsum((self.ExpXw * self.X[:, j])[::-1])[::-1]
        reverse_sum_ExpXw_X_squared = np.cumsum((self.ExpXw * self.X_squared[:, j])[::-1])[::-1]

        term1 = np.sum(self.same_time_same_event_counts * reverse_sum_ExpXw_X_squared[self.same_time_first_event_indices] / reverse_sum_ExpXw[self.same_time_first_event_indices])
        term2 = np.sum(self.same_time_same_event_counts * (reverse_sum_ExpXw_X[self.same_time_first_event_indices] / reverse_sum_ExpXw[self.same_time_first_event_indices]) ** 2)

        return term1 - term2 + self.twolambda2

    def optimize_coordinate_j_per_iteration(self, j):
        grad_j = self.get_feature_gradient_at_coordinate_j(j)
        hessian_j = self.get_feature_hessian_at_coordinate_j(j)

        # hessian_j_slow = self.get_feature_hessian_at_coordinate_j_slow(j)
        # print(f"hessian diff: {hessian_j - hessian_j_slow}")

        L3_j = self.featurewise_diagCubicUpperBound[j]

        dw_j = soft_thresholding_operator_cubic_shifted(grad_j, hessian_j, L3_j, self.w[j], self.lambda1)
        self.w[j] += dw_j
        self.ExpXw *= np.exp(self.X[:, j] * dw_j)

class coxModelElasticNet_newton_cubic_surrogate_accelerated(coxModelElasticNet_newton_cubic_surrogate):
    def __init__(
            self,
            X,
            y_time,
            y_event,
            verbose=False,
            lambda1=0.0,
            lambda2=1e-3,
    ):
        if lambda1 != 0:
            raise ValueError("lambda1 should be 0 for newton_cubic_surrogate_accelerated")

        super().__init__(X=X, y_time=y_time, y_event=y_event, verbose=verbose, lambda1=lambda1, lambda2=lambda2)

        self.theta = 1

        self.u = self.w.copy()
        self.v = self.w.copy()

    def optimize_coordinate_j_per_iteration(self, j):
        dw_j = (1-self.theta) * self.u[j] + self.theta * self.v[j] - self.w[j]
        print(f"dw_j: {dw_j}")
        self.w[j] += dw_j
        self.ExpXw *= np.exp(self.X[:, j].dot(dw_j))

        grad_j = self.get_feature_gradient_at_coordinate_j(j)
        hessian_j = self.get_feature_hessian_at_coordinate_j(j)
        L3_j = self.featurewise_diagCubicUpperBound[j]

        sign_grad_j = math.copysign(1, grad_j)
        dw_j = sign_grad_j * (hessian_j - math.sqrt(hessian_j **2 + 2 * L3_j * abs(grad_j))) / L3_j

        dw_j_thresholded = soft_thresholding_operator_cubic_shifted(grad_j, hessian_j, L3_j, self.w[j], self.lambda1)
        if dw_j != dw_j_thresholded:
            print(f"dw_j: {dw_j}, dw_j_thresholded: {dw_j_thresholded}")
            sys.exit()
        print(f"dw_j: {dw_j}, dw_j_thresholded: {dw_j_thresholded}")

        self.u[j] = self.w[j] + dw_j
        # self.v[j] = self.v[j] + 1 / (self.p * self.theta) * dw_j
        self.v[j] = self.w[j] + dw_j

        if self.u[j] != self.v[j]:
            print("error!")
            sys.exit()

        self.theta = (-self.theta ** 2 + self.theta * math.sqrt(self.theta ** 2 + 4)) / 2

    def optimize_all_coordinates_per_iteration(self):
        for j in range(self.p):
            self.optimize_coordinate_j_per_iteration(j)
            if j == 10:
                sys.exit()

def soft_thresholding_operator_cubic_shifted(a, b, c, d, lambda1):
    # if lambda1 == 0:
    #     discriminant = b ** 2 + 2 * c * abs(a)
    #     value = (b - math.sqrt(discriminant)) / c
    #     return math.copysign(1, a) * value # np.sign(a) * value

    sign_d = math.copysign(1, d)
    if sign_d * a + lambda1 == 0:
        return 0
    elif sign_d * a + lambda1 < 0:
        discriminant = b ** 2 - sign_d * 2 * a * c - 2 * c * lambda1
        return sign_d * (-b + math.sqrt(discriminant)) / c
    else:
        tmp_to_comparison_with_lambda1 = sign_d * (a - b * d) - 0.5 * c * d ** 2
        if tmp_to_comparison_with_lambda1 > lambda1:
            discriminant = b ** 2 + sign_d * (2 * a * c) - 2 * c * lambda1
            return sign_d * (b - math.sqrt(discriminant)) / c
        elif tmp_to_comparison_with_lambda1 < -lambda1:
            discriminant = b ** 2 + sign_d * (2 * a * c) + 2 * c * lambda1
            return sign_d * (b - math.sqrt(discriminant)) / c
        else:
            return -d
