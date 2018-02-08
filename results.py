#! /usr/bin/env python -u
# coding=utf-8
import numpy as np

from oracle import TimeOracle
from utils import Efficiency

__author__ = 'xl'

import re
import pickle
import os
from collections import OrderedDict


class ResultAnalyser:
    def __init__(self, experiment_result):
        self.effs = []
        self.all_effs = []
        self.worker_devices = []

        self._update(experiment_result)

    def _update(self, experiment_result):
        if len(experiment_result.metadata) == 0:
            return
        self.worker_devices = \
            list(set([a.device for a in experiment_result.metadata[-1].step_stats.dev_stats if "worker" in a.device]))

        for m in experiment_result.metadata:
            effs_ = [Efficiency(m, device_search=d) for d in self.worker_devices]
            self.effs.append(effs_)
            self.all_effs += effs_

    def get_a(self):
        a = [e.a for e in self.all_effs]
        return np.mean([min(a), max(a)])

