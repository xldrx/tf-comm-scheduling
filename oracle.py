#! /usr/bin/env python -u
# coding=utf-8
import json
import re
import tensorflow as tf

__author__ = 'Sayed Hadi Hashemi'


class TimeOracle:
    def __init__(self, scope=None):
        self._time = {}
        self._scope = scope

    @classmethod
    def load(cls, filename, scope):
        with open(filename, "r") as fp:
            _time = json.load(fp)
        oracle = cls()
        oracle._time = _time
        oracle._scope = scope
        return oracle

    def save(self, filename):
        with open(filename, "w") as fp:
            json.dump(self._time, fp, indent=2, sort_keys=True)

    def update(self, metadata):
        metadata_copy = tf.RunMetadata()
        metadata_copy.CopyFrom(metadata)
        all_ops = []
        for device in [d for d in metadata_copy.step_stats.dev_stats if "worker" in d.device]:
            all_ops += device.node_stats

        last_end = 0
        for op in sorted(all_ops, key=lambda a: a.all_start_micros + a.all_end_rel_micros):
            if op.node_name == "RecvTensor":
                op_start = max(op.all_start_micros, last_end - 1)
                op_end = op.all_start_micros + op.all_end_rel_micros
                op.all_start_micros = op_start
                op.op_end_rel_micros = op_end - op_start
                op.all_end_rel_micros = op_end - op_start
                last_end = op_end
                op_name = self.recvop_name(op)
            else:
                op_name = self.remove_prefix(op.node_name)
            if op_name:
                self._time[op_name] = min(op.all_end_rel_micros, self._time.get(op_name, op.all_end_rel_micros))

    def query(self, name):
        fixed_name = self.remove_prefix(name)
        if fixed_name in self._time:
            return self._time[fixed_name]

        recv_name = "recv:{}".format(fixed_name)
        if recv_name in self._time:
            return self._time[recv_name]
        else:
            return None

    def remove_prefix(self, name):
        # 2
        op_name = re.findall("^{scope}(?:_\d+)?/{scope}/(.*)$".format(scope=self._scope), name)
        if op_name:
            return "//" + op_name[0]
        # 1
        op_name = re.findall("^{scope}(?:_\d+)?/(.*)$".format(scope=self._scope), name)
        if op_name:
            return "/" + op_name[0]
        # 0
        return name

    def recvop_name(self, op):
        # TODO(xldrx): extends the support beyond MR+PS.
        tensorname = re.findall(".* edge_\d+_(.+)/read from .*", op.timeline_label)
        if tensorname:
            op_name = "recv:{}".format(self.remove_prefix(tensorname[0]))
            return op_name
        return None
