#!/usr/bin/env
import torch
from metrics.metric import Metric


class F1Measure(Metric):
    """
    Implementation of f1 metric
    """

    def __init__(self):
        super(F1Measure, self).__init__()
        self._tp, self._fp, self._fn, self._tn = 0, 0, 0, 0

    def update(self, pred, target):
        self._tp += (target + pred == 2).sum()
        self._fp += (pred - target == 1).sum()
        self._fn += (pred - target == -1).sum()
        self._tn += (target + pred == 0).sum()

    def flush(self):
        self._tp, self._fp, self._fn = 0, 0, 0

    def compute(self):
        print(f'TP {self._tp}', f'FP {self._fp}')
        print(f'FN {self._fn}', f'TN {self._tn}')
        print(f'Recall: {torch.true_divide(self._tp, (self._tp + self._fn))}')
        print(f'Precision: {torch.true_divide(self._tp, (self._tp + self._fp))}')
        print(f'Specificity: {torch.true_divide(self._tn, (self._tn + self._fp))} ')
        print(f'F1 measure: {torch.true_divide(self._tp, self._tp + .5 * (self._fp + self._fn))}')
        return torch.true_divide(self._tp, self._tp + .5 * (self._fp + self._fn))
