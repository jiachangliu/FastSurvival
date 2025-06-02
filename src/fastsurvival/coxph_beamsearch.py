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

from sksurv.linear_model import CoxPHSurvivalAnalysis as sksurv_CoxPHSurvivalAnalysis
import math



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

        self.samplewise_diagHessianUpperBound = np.zeros((self.n,))

        self.samplewise_diagHessianUpperBound[self.same_time_first_event_indices] = self.same_time_same_event_counts * 0.25
        self.samplewise_diagHessianUpperBound = np.cumsum(self.samplewise_diagHessianUpperBound)

        self.featurewise_diagHessianUpperBound = np.zeros((self.p,))
        cummin_X = np.minimum.accumulate(self.X[::-1], axis=0)[::-1]
        cummax_X = np.maximum.accumulate(self.X[::-1], axis=0)[::-1]

        self.featurewise_diagHessianUpperBound = 0.25 * np.sum(self.same_time_same_event_counts.reshape(-1, 1) * (cummax_X[self.same_time_first_event_indices] - cummin_X[self.same_time_first_event_indices])** 2, axis=0)

        self.featurewise_diagCubicUpperBound = 1/(6 * math.sqrt(3)) * np.sum(self.same_time_same_event_counts.reshape(-1, 1) * (cummax_X[self.same_time_first_event_indices] - cummin_X[self.same_time_first_event_indices])** 3, axis=0)

    
    def reset_coef(self, w):
        self.w = w

    def compute_loss(self, ExpXw, w):
        first_term = np.sum(np.log(ExpXw)[self.event_1_mask])
        reverse_sum_ExpyXw = np.cumsum(ExpXw[::-1])[::-1]
        second_term = np.sum(np.log(reverse_sum_ExpyXw[self.same_time_first_event_indices]) * self.same_time_same_event_counts)

        return -first_term + second_term + self.lambda2 * np.sum(w ** 2)
    
    # def finetune_on_current_support(self, X_on_supp_mask, X_squared_on_supp_mask=None, w_on_supp_mask=None, max_iter=1000):
    #     sksurv_model = sksurv_CoxPHSurvivalAnalysis()
    #     sksurv_model.fit(X_on_supp_mask, self.y_sksurv)

    #     w_on_supp_mask = sksurv_model.coef_
    #     Xw_on_supp_mask = X_on_supp_mask.dot(w_on_supp_mask)
    #     ExpXw_on_supp_mask = np.exp(Xw_on_supp_mask)
    #     loss = np.sum(np.log(np.cumsum(ExpXw_on_supp_mask[::-1])[::-1][self.same_time_first_event_indices]) * self.same_time_same_event_counts) - np.sum(Xw_on_supp_mask[self.event_1_mask])

    #     # print(f"after finetuning, w_on_supp_mask is {w_on_supp_mask}, loss is {loss}")
    #     return w_on_supp_mask, ExpXw_on_supp_mask, loss

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
        samplewise_diagHessian = self.samplewise_diagHessianUpperBound

        return samplewise_gradient, samplewise_diagHessian

    def finetune_on_current_support(self, X_on_supp_mask, X_squared_on_supp_mask=None, w_on_supp_mask=None, featurewise_diagHessianUpperBound=None, max_iter=10000):
        if X_squared_on_supp_mask is None:
            X_squared_on_supp_mask = X_on_supp_mask ** 2

        if w_on_supp_mask is None:
            w_on_supp_mask = np.zeros(X_on_supp_mask.shape[1])
        
        Xw_on_supp_mask = X_on_supp_mask.dot(w_on_supp_mask)
        ExpXw_on_supp_mask = np.exp(Xw_on_supp_mask)
        current_loss = self.compute_loss(ExpXw_on_supp_mask, w_on_supp_mask)

        # print(f"before finetuning, current_loss is {current_loss}")
        for iter in range(max_iter):
            for j in range(X_on_supp_mask.shape[1]):
                samplewise_gradient, samplewise_diagHessian = self.compute_samplewise_gradient_and_hessian(ExpXw_on_supp_mask)
                # delta_w_j = - (np.sum(samplewise_gradient * X_on_supp_mask[:, j]) + self.twolambda2 * w_on_supp_mask[j]) / (np.sum(samplewise_diagHessian * X_squared_on_supp_mask[:, j]) + self.twolambda2)

                delta_w_j = - (np.sum(samplewise_gradient * X_on_supp_mask[:, j]) + self.twolambda2 * w_on_supp_mask[j]) / (featurewise_diagHessianUpperBound[j] + self.twolambda2)

                # if np.max(delta_w_j * X_on_supp_mask[:, j]) > 2700:
                #     print(delta_w_j)
                #     print(j)
                #     print(np.max(X_on_supp_mask, axis=0))
                #     print(np.min(X_on_supp_mask, axis=0))
                #     print(f"max(delta_w_j * X_on_supp_mask[:, j]) is {np.max(delta_w_j * X_on_supp_mask[:, j])}")
                #     print(f"min(delta_w_j * X_on_supp_mask[:, j]) is {np.min(delta_w_j * X_on_supp_mask[:, j])}")
                #     print(X_on_supp_mask[:, j])
                #     print(w_on_supp_mask[j])
                #     print(w_on_supp_mask)
                #     print("why 2700?")
                #     print(np.sum(samplewise_gradient * X_on_supp_mask[:, j]))
                #     print(np.sum(samplewise_diagHessian * X_squared_on_supp_mask[:, j]))
                #     print(samplewise_gradient)
                #     print(samplewise_diagHessian)
                #     print(sum(samplewise_diagHessian > 0))
                #     sys.exit()
                ExpXw_on_supp_mask = ExpXw_on_supp_mask * np.exp(delta_w_j * X_on_supp_mask[:, j])
                w_on_supp_mask[j] = w_on_supp_mask[j] + delta_w_j
            # tmp_loss = self.compute_loss(ExpXw_on_supp_mask, w_on_supp_mask)
            # print(f"iter {iter}, loss is {tmp_loss}")
            if iter % 10 == 9:
                new_loss = self.compute_loss(ExpXw_on_supp_mask, w_on_supp_mask)
                if (current_loss - new_loss) / new_loss < 1e-6:
                    print(f"early break at iter {iter}")
                    break
                current_loss = new_loss
            if iter == max_iter - 1:
                print(f"max_iter reached")

        loss = self.compute_loss(ExpXw_on_supp_mask, w_on_supp_mask)
        # print(f"after finetuning, loss is {loss}")
        return w_on_supp_mask, ExpXw_on_supp_mask, loss

class sparseCoxModel_big_p(coxModel_unnormalized_big_p):
    def __init__(
        self,
        X,
        y_time,
        y_event,
        parent_size=5,
        child_size=5,
        allowed_supp_mask=None,
        verbose=False,
        lambda2=1e-3,
    ):
        super().__init__(X=X, y_time=y_time, y_event=y_event)

        self.parent_size = parent_size
        self.child_size = child_size


        if allowed_supp_mask is None:
            self.allowed_supp_mask = np.ones((self.p,), dtype=bool)
        else:
            self.allowed_supp_mask = allowed_supp_mask

        self.saved_solution = {}
        supp_mask_beginning = np.zeros((self.p,), dtype=bool)
        tmp_support_str = supp_mask_beginning.tostring()

        w_beginning = np.zeros((self.p,))
        ExpXw_beginning = np.exp(np.zeros((self.n,)))
        loss_beginning = self.compute_loss(ExpXw_beginning, w_beginning)
        self.saved_solution[tmp_support_str] = (None, ExpXw_beginning, loss_beginning)

        self.verbose = verbose
        self.lambda2 = lambda2
        self.twolambda2 = 2 * lambda2
    
    def refine_coef_through_random_delete(self, w, num_iter=10, delete_ratio=0.3):
        k = np.count_nonzero(w)
        print(f"k is {k}")
        print(f"default delete_ratio is {delete_ratio}")
        tmp_support_mask = w != 0
        tmp_support_mask_str = tmp_support_mask.tostring()

        if tmp_support_mask_str not in self.saved_solution:
            # w_on_tmp_support_mask, ExpXw_tmp, loss = self.finetune_on_current_support(self.X[:, tmp_support_mask])
            w_on_tmp_support_mask, ExpXw_tmp, loss = self.finetune_on_current_support(self.X[:, tmp_support_mask], self.X_squared[:, tmp_support_mask], w[tmp_support_mask], self.featurewise_diagHessianUpperBound[tmp_support_mask])
            self.saved_solution[tmp_support_mask_str] = (w_on_tmp_support_mask, ExpXw_tmp, loss)

        num_delete = min(k, max(int(k * delete_ratio), 1))

        best_w_support, _, best_loss = self.saved_solution[tmp_support_mask_str]
        best_w = np.zeros((self.p,))
        best_w[tmp_support_mask] = best_w_support

        print(f"at the beginning, best_loss is {best_loss}, support indices are {tmp_support_mask.nonzero()[0]}")

        for _ in range(num_iter):
            print()
            tmp_support_mask = best_w != 0
            delete_indices = np.random.choice(np.where(tmp_support_mask)[0], num_delete, replace=False)
            print(f"delete_indices are {delete_indices}")
            w = best_w.copy()
            w[delete_indices] = 0

            self.reset_coef(w)
            w_new = self.get_sparse_sol_via_OMP(k)
            loss = self.compute_loss(np.exp(self.X @ w_new), w_new)
            print(f"loss is {loss}, support indices are {w_new.nonzero()[0]}")

            if loss < best_loss:
                best_loss = loss
                best_w = w_new.copy()
                print(f"!!! found better loss, best_loss is {best_loss}, support indices are {np.where(best_w != 0)[0]}")
        
        return best_w


    def get_sparse_sol_via_OMP(self, k):
        self.heaps = [[] for _ in range(k + 1)]

        # data structure of the heap will be (-loss, main_indices_mask_str, r)
        tmp_support_mask = self.w != 0
        tmp_support_mask_str = tmp_support_mask.tostring()
        support_size = np.count_nonzero(tmp_support_mask)

        if tmp_support_mask_str not in self.saved_solution:
            # w_on_tmp_support_mask, ExpXw_tmp, loss = self.finetune_on_current_support(self.X[:, tmp_support_mask])
            w_on_tmp_support_mask, ExpXw_tmp, loss = self.finetune_on_current_support(self.X[:, tmp_support_mask], self.X_squared[:, tmp_support_mask], self.w[tmp_support_mask], self.featurewise_diagHessianUpperBound[tmp_support_mask])
            self.saved_solution[tmp_support_mask_str] = (w_on_tmp_support_mask, ExpXw_tmp, loss)

        w_on_tmp_support_mask, ExpXw_tmp, loss = self.saved_solution[tmp_support_mask_str]
        heappush(self.heaps[support_size], (-loss, tmp_support_mask_str, ExpXw_tmp))


        # ExpXw = np.exp(self.X @ self.w)
        # loss_beginning = self.compute_loss(ExpXw)
        # print(f'loss_beginning is {loss_beginning}')
        # print(f"support_size is {support_size}")
        # heappush(self.heaps[support_size], (-loss_beginning, tmp_support_mask_str, ExpXw))

        # best_loss = loss_beginning
        # best_mask = np.zeros(self.p, dtype=bool)
        support_size = support_size + 1
        print(f"k is {k}")

        while support_size <= k:
            print(f'support_size is {support_size}')

            for (neg_loss, support_mask_str, ExpXw) in self.heaps[support_size-1]:
                # print(f"before parallel CD, loss is {-neg_loss}")
                support_mask = np.fromstring(support_mask_str, dtype=bool)
                tmp_allowed_supp_mask = self.allowed_supp_mask & ~support_mask

                ExpXw_on_tmp_allowed_supp = np.tile(ExpXw.reshape(-1, 1), (1, sum(tmp_allowed_supp_mask)))

                reverse_sum_ExpXw_on_tmp_allowed_supp = np.cumsum(ExpXw_on_tmp_allowed_supp[::-1], axis=0)[::-1]
                loss_on_tmp_allowed_supp = np.sum(np.log(reverse_sum_ExpXw_on_tmp_allowed_supp[self.same_time_first_event_indices]) * self.same_time_same_event_counts.reshape(-1, 1), axis=0)
                loss_on_tmp_allowed_supp -= np.sum(np.log(ExpXw_on_tmp_allowed_supp[self.event_1_mask]), axis=0)
                current_loss = loss_on_tmp_allowed_supp
                # print(f'before parallel CD, loss_on_tmp_allowed_supp is {loss_on_tmp_allowed_supp}')


                w_j_on_tmp_allowed_supp = np.zeros(sum(tmp_allowed_supp_mask))

                for _ in range(100):
                    
                    reverse_sum_ExpXw = np.cumsum(ExpXw_on_tmp_allowed_supp[::-1], axis=0)[::-1]
                    common_term = np.zeros((self.n, sum(tmp_allowed_supp_mask)))
                    common_term[self.same_time_first_event_indices] = self.same_time_same_event_counts.reshape(-1, 1) / reverse_sum_ExpXw[self.same_time_first_event_indices]
                    common_term = np.cumsum(common_term, axis=0)
                    common_term *= ExpXw_on_tmp_allowed_supp

                    samplewise_gradient = common_term - self.event_1_mask.reshape(-1, 1)

                    second_term = np.zeros((self.n, sum(tmp_allowed_supp_mask)))
                    second_term[self.same_time_first_event_indices] = self.same_time_same_event_counts.reshape(-1, 1) / reverse_sum_ExpXw[self.same_time_first_event_indices] ** 2
                    second_term = np.cumsum(second_term, axis=0)
                    second_term *= ExpXw_on_tmp_allowed_supp ** 2

                    # Method 1
                    # samplewise_diagHessian = common_term - second_term

                    # Method 2
                    # samplewise_diagHessian = common_term

                    # Method 3
                    # samplewise_diagHessian = self.samplewise_diagHessianUpperBound.reshape(-1, 1)
                    # delta_w = - (np.sum(samplewise_gradient * self.X[:, tmp_allowed_supp_mask], axis=0) + self.twolambda2 * w_j_on_tmp_allowed_supp) / (np.sum(samplewise_diagHessian * self.X_squared[:, tmp_allowed_supp_mask], axis=0) + self.twolambda2)

                    # Method 4
                    delta_w = - (np.sum(samplewise_gradient * self.X[:, tmp_allowed_supp_mask], axis=0) + self.twolambda2 * w_j_on_tmp_allowed_supp) / (self.featurewise_diagHessianUpperBound[tmp_allowed_supp_mask] + self.twolambda2)

                    # # Method 5
                    # reverse_sum_ExpXw_X_on_tmp_allowed_supp = np.cumsum((ExpXw_on_tmp_allowed_supp * self.X[:, tmp_allowed_supp_mask])[::-1], axis=0)[::-1]
                    # reverse_sum_ExpXw_X_squared_on_tmp_allowed_supp = np.cumsum((ExpXw_on_tmp_allowed_supp * self.X_squared[:, tmp_allowed_supp_mask])[::-1], axis=0)[::-1]

                    # term1 = np.sum(self.same_time_same_event_counts.reshape(-1, 1) * reverse_sum_ExpXw_X_squared_on_tmp_allowed_supp[self.same_time_first_event_indices] / reverse_sum_ExpXw_on_tmp_allowed_supp[self.same_time_first_event_indices], axis=0)
                    # term2 = np.sum(self.same_time_same_event_counts.reshape(-1, 1) * (reverse_sum_ExpXw_X_on_tmp_allowed_supp[self.same_time_first_event_indices] / reverse_sum_ExpXw_on_tmp_allowed_supp[self.same_time_first_event_indices])**2, axis=0)

                    # featurewise_diagHessian = term1 - term2 + self.twolambda2
                    # featurewise_gradient = np.sum(self.X[:, tmp_allowed_supp_mask] * samplewise_gradient, axis=0) + self.twolambda2 * w_j_on_tmp_allowed_supp

                    # delta_w = np.sign(featurewise_gradient[tmp_allowed_supp_mask]) * (featurewise_diagHessian[tmp_allowed_supp_mask] - np.sqrt(featurewise_diagHessian[tmp_allowed_supp_mask] **2 + 2 * np.abs(featurewise_gradient[tmp_allowed_supp_mask]) * self.featurewise_diagCubicUpperBound[tmp_allowed_supp_mask])) / self.featurewise_diagCubicUpperBound[tmp_allowed_supp_mask]
                    



                    ExpXw_on_tmp_allowed_supp = ExpXw_on_tmp_allowed_supp * np.exp(delta_w.reshape(1, -1) * self.X[:, tmp_allowed_supp_mask])
                    w_j_on_tmp_allowed_supp = w_j_on_tmp_allowed_supp + delta_w
                
                    #################################################################
                    if self.verbose:
                        reverse_sum_ExpXw_on_tmp_allowed_supp = np.cumsum(ExpXw_on_tmp_allowed_supp[::-1], axis=0)[::-1]
                        loss_on_tmp_allowed_supp = np.sum(np.log(reverse_sum_ExpXw_on_tmp_allowed_supp[self.same_time_first_event_indices]) * self.same_time_same_event_counts.reshape(-1, 1), axis=0)
                        loss_on_tmp_allowed_supp -= np.sum(np.log(ExpXw_on_tmp_allowed_supp[self.event_1_mask]), axis=0)
                        print(f'loss_on_tmp_allowed_supp is {loss_on_tmp_allowed_supp}')
                        if np.sum(loss_on_tmp_allowed_supp > current_loss + 1e-8) > 0:
                            violation_indices = np.where(loss_on_tmp_allowed_supp > current_loss)[0]
                            print(f"violated indices are {violation_indices}")
                            print(f"loss increase is {loss_on_tmp_allowed_supp[violation_indices] - current_loss[violation_indices]}")

                            raise ValueError(f"Error: some loss_on_tmp_allowed_supp is not monotonically decreasing")
                        current_loss = loss_on_tmp_allowed_supp
                        print()
                    #################################################################
                
                reverse_sum_ExpXw_on_tmp_allowed_supp = np.cumsum(ExpXw_on_tmp_allowed_supp[::-1], axis=0)[::-1]
                loss_on_tmp_allowed_supp = np.sum(np.log(reverse_sum_ExpXw_on_tmp_allowed_supp[self.same_time_first_event_indices]) * self.same_time_same_event_counts.reshape(-1, 1), axis=0)
                loss_on_tmp_allowed_supp -= np.sum(np.log(ExpXw_on_tmp_allowed_supp[self.event_1_mask]), axis=0)
                # if np.sum(loss_on_tmp_allowed_supp >= current_loss) > 0:
                #     print(f"loss_on_tmp_allowed_supp is {loss_on_tmp_allowed_supp}")
                #     raise ValueError(f"Error: some loss_on_tmp_allowed_supp is not decreasing")
                # print(f'after parallel CD, loss_on_tmp_allowed_supp is {loss_on_tmp_allowed_supp}')

                tmp_allowed_supp_indices = np.where(tmp_allowed_supp_mask)[0]
                sorted_indices_on_tmp_allowed_supp_child_size = np.argsort(loss_on_tmp_allowed_supp)[:self.child_size]
                top_candidate_indices = tmp_allowed_supp_indices[sorted_indices_on_tmp_allowed_supp_child_size]
                # print(f"loss_on_tmp_allowed_supp are {loss_on_tmp_allowed_supp[np.argsort(loss_on_tmp_allowed_supp)[:self.child_size]]}")
                # print(f"top_candidate_indices are {top_candidate_indices}")

                for j, top_candidate_ind in enumerate(top_candidate_indices):
                    # print(f"\nj is {j}, top_candidate_ind is {top_candidate_ind}")
                    # print(f"\nj is {j}, top_candidate_ind is {top_candidate_ind}, support_mask is {support_mask}")
                    tmp_support_mask = support_mask.copy()
                    tmp_support_mask_str = tmp_support_mask.tostring()

                    w_on_tmp_support_mask, _, _ = self.saved_solution[tmp_support_mask_str]
                    # print("yoyoyo1", self.saved_solution[tmp_support_mask_str])


                    w = np.zeros((self.p,))
                    # print(f"w_on_tmp_support_mask is {w_on_tmp_support_mask}")
                    # print(f"tmp_support_mask is {tmp_support_mask}")
                    if w_on_tmp_support_mask is not None:
                        if len(w_on_tmp_support_mask) != sum(tmp_support_mask):
                            raise ValueError(f"Error1: len(w_on_tmp_support_mask) is {len(w_on_tmp_support_mask)}, sum(tmp_support_mask) is {sum(tmp_support_mask)}")
                    w[tmp_support_mask] = w_on_tmp_support_mask
                    # print("yoyoyo2", self.saved_solution[tmp_support_mask_str])

                    tmp_support_mask[top_candidate_ind] = True
                    tmp_support_mask_str = tmp_support_mask.tostring()
                    w[top_candidate_ind] = w_j_on_tmp_allowed_supp[sorted_indices_on_tmp_allowed_supp_child_size[j]]
                    # print(f"after setting, tmp_support_mask is {tmp_support_mask}")

                    if tmp_support_mask_str not in self.saved_solution:
                        # w_on_tmp_support_mask, ExpXw_tmp, loss = self.finetune_on_current_support(self.X[:, tmp_support_mask])

                        w_on_tmp_support_mask, ExpXw_tmp, loss = self.finetune_on_current_support(self.X[:, tmp_support_mask], self.X_squared[:, tmp_support_mask], w[tmp_support_mask], self.featurewise_diagHessianUpperBound[tmp_support_mask])
                        if len(w_on_tmp_support_mask) != sum(tmp_support_mask):
                            raise ValueError(f"Error2: len(w_on_tmp_support_mask) is {len(w_on_tmp_support_mask)}, sum(tmp_support_mask) is {sum(tmp_support_mask)}")
                        # print(f'loss is {loss}')
                        self.saved_solution[tmp_support_mask_str] = (w_on_tmp_support_mask, ExpXw_tmp, loss)

                        heappush(self.heaps[support_size], (-loss, tmp_support_mask_str, ExpXw_tmp))
                    else:
                        _, ExpXw_tmp, loss = self.saved_solution[tmp_support_mask_str]
                        heappush(self.heaps[support_size], (-loss, tmp_support_mask_str, ExpXw_tmp))
                
                # print()

            while len(self.heaps[support_size]) > self.parent_size:
                heappop(self.heaps[support_size])
            
            support_size += 1
        
        print("best solution:")
        print(f"last heap has {len(self.heaps[k])} solutions")
        while len(self.heaps[k]) > 1:
            heappop(self.heaps[k])
        neg_loss, support_mask_str, ExpXw = heappop(self.heaps[k])
        support_mask = np.fromstring(support_mask_str, dtype=bool)
        print(support_mask.nonzero()[0])
        print(-neg_loss)
        print(f'computed loss is {self.compute_loss(ExpXw, self.saved_solution[support_mask_str][0])}')

        w = np.zeros((self.p,))
        w[support_mask] = self.saved_solution[support_mask_str][0]

        return w