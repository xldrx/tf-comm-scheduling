#! /usr/bin/env python -u
# coding=utf-8
from functools import lru_cache
import utils
import math
import tensorflow as tf

__author__ = 'Sayed Hadi Hashemi'

class BaseOrdering:
    def __init__(self, target_node):
        self._target = target_node
        self._find_comm_dependencies()
        self._seperate_comp_comm()

    @staticmethod
    def _is_recv(op):
        if op.op.name.endswith("/read"):
            return op.op.name[:-5]
        else:
            return None

    def _find_comm_dependencies(self):
        stack = [self._target]
        processed = {}
        while stack:
            op = stack.pop()
            if processed.get(op, False):
                deps = set()
                for input_op in op.op.inputs:
                    deps.update(input_op.deps)
                op.deps = deps
            else:
                if self._is_recv(op):
                    op.deps = {op}
                else:
                    stack.append(op)
                    for input_op in op.op.inputs:
                        stack.append(input_op)
                processed[op] = True

    def _seperate_comp_comm(self):
        self._comp_ops = []
        self._comm_ops = []
        stack = [self._target]
        processed = {}
        while stack:
            op = stack.pop()
            if op not in processed:
                if self._is_recv(op):
                    self._comm_ops.append(op)
                else:
                    self._comp_ops.append(op)
                    for input_op in op.op.inputs:
                        stack.append(input_op)
                processed[op] = True

    def _update_properties(self, outstanding_comm_ops):

        for op in self._comm_ops:
            op.P = 0
            op.Mp = math.inf
            op.M = self._get_time(op)

        for op in self._comp_ops:
            op_deps = op.deps.intersection(outstanding_comm_ops)
            if len(op_deps) == 1:
                for read in op_deps:
                    read.P += self._get_time(op)
            elif len(op_deps) > 1:
                op_M = sum(self._get_time(r) for r in op_deps)
                for read in op_deps:
                    read.Mp = min(op_M, read.Mp)

    def _get_time(self, op):
        raise NotImplementedError()

class TAO(BaseOrdering):
    def __init__(self, target_node, time_oracle):
        super().__init__(target_node)
        self._time_oracle = time_oracle

    @staticmethod
    def _comparator(op1, op2):
        a = min(op2.P, op1.M)
        b = min(op1.P, op2.M)

        if a != b:
            return a < b
        else:
            return op1.Mp < op2.Mp

    def _get_time(self, op):
        if self._is_recv(op):
            op_name = self._is_recv(op)
        else:
            op_name = op.op.name

        time = self._time_oracle.query(op_name)
        if time:
            return time
        else:
            print("// >>> Error (Server-Client version mismatch?): {}".format(op_name))
            return 10

    @lru_cache()
    def get_priorities(self):
        outstanding_comm_ops = list(self._comm_ops)
        priorities = []
        counter = 0
        while outstanding_comm_ops:
            self._update_properties(outstanding_comm_ops)
            outstanding_comm_ops.sort(key=utils.cmp_to_key(self._comparator))
            priorities.append((counter, outstanding_comm_ops[0]))
            counter += 1
            del outstanding_comm_ops[0]
        return priorities


class TIO(BaseOrdering):
    @lru_cache()
    def _get_time(self, op):
        if self._is_recv(op):
            return 1
        else:
            return 0

    @lru_cache()
    def get_priorities(self):
        outstanding_comm_ops = list(self._comm_ops)
        self._update_properties(outstanding_comm_ops)
        priorities = []
        counter = 0
        last_counter = -1
        last_Mp = -1
        for op in sorted(outstanding_comm_ops, key=lambda recv_op: recv_op.Mp):
            if last_Mp < op.Mp:
                last_counter = counter
                last_Mp = op.Mp
            priorities.append((last_counter, op))
            counter += 1
        return priorities
