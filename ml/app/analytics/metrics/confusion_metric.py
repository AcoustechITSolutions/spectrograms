#!/usr/bin/env
from metrics.metric import Metric


class ConfusionMetric(Metric):
    """
    Implementation of f1 metric
    """

    def __init__(self):
        super(ConfusionMetric, self).__init__()
        self._list = []

    def update(self, pred, target):
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()

        matrix = {2: 'tp', 0: 'tn', 1: 'fp', -1: 'fn'}
        for idx in range(len(pred)):
            sum_pt, sub_pt = pred[idx] + target[idx], pred[idx] - target[idx]
            self._list.append(matrix[sum_pt if sum_pt == 2 or 0 else sub_pt])

    def flush(self):
        self._list = []

    def compute(self):
        return self._list
