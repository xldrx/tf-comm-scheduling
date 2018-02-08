#! /usr/bin/env python -u
# coding=utf-8
import json
import re
import time
import tensorflow as tf

__author__ = 'Sayed Hadi Hashemi'


def cmp_to_key(mycmp):
    'Convert a cmp= function into a key= function'

    class K:
        def __init__(self, obj, *args):
            self.obj = obj

        def __lt__(self, other):
            return mycmp(self.obj, other.obj)

        def __gt__(self, other):
            return not mycmp(self.obj, other.obj)

    return K


class Timer:
    def start_timer(self):
        self.start = time.time()

    def __enter__(self):
        self.start_timer()
        return self

    def __exit__(self, *args):
        self.stop_timer()

    def stop_timer(self):
        self.end = time.time()

    def elapsed(self):
        return self.end - self.start


class Timeline:
    def __enter__(self):
        self.run_metadata = tf.RunMetadata()
        self.options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
        return self

    def kwargs(self):
        return dict(run_metadata=self.run_metadata, options=self.options)

    def __exit__(self, *args):
        pass


class Tracker:
    def __init__(self):
        self._utilization = 0
        self.ops = []
        self._start = 0
        self._end = 0

    def add_op(self, op):
        if not self.ops:
            self._start = op.all_start_micros
        op_end = op.all_start_micros + op.op_end_rel_micros
        self._utilization += max(0,
                                 op.all_start_micros + op.op_end_rel_micros - max(self._end, op.all_start_micros))

        self._start = min(op.all_start_micros, self._start)
        self._end = max(self._end, op_end)
        self.ops.append(op)

    def makespan(self):
        return self._end - self._start

    def utilization(self):
        return self._utilization


class Efficiency:
    def __init__(self, run_metadata, device_search="worker"):
        self.comm = Tracker()
        self.comp = Tracker()
        all_ops = []
        for device in [d for d in run_metadata.step_stats.dev_stats if device_search in d.device]:
            all_ops += device.node_stats
        for op in sorted(all_ops, key=lambda a: a.all_start_micros):
            if op.node_name == "RecvTensor":
                self.comm.add_op(op)
            else:
                self.comp.add_op(op)
        self.U = max(self.comm._end, self.comp._end) - min(self.comp._start, self.comp._start)
        self.cost_max = self.comm.utilization() + self.comp.utilization()
        self.cost_min = max(self.comm.utilization(), self.comp.utilization())
        self.E = (self.cost_max - self.U) / (self.cost_max - self.cost_min) \
            if (self.cost_max - self.cost_min) != 0 else -1
        self.S = (self.cost_max - self.cost_min) / self.cost_min if self.cost_min != 0 else -1
        self.a = self.comm.utilization() / self.comp.utilization() if self.comp.utilization() != 0 else -1
        self.P = self.comp.utilization()
        self.M = self.comm.utilization()

    def __str__(self):
        return "E: {:0.2f}, S: {:0.2f}, a: {:0.2f} E*S: {:0.2f} M: {:0.0f} P: {:0.0f} U: {:0.0f}".format(
            self.E, self.S, self.a, self.E * self.S, self.M / 1000, self.P / 1000, self.U / 1000
        )


try:
    log_progress = lambda x: x
    from tqdm import tqdm

    log_progress = tqdm
except ImportError:
    pass
