#! /usr/bin/env python -u
# coding=utf-8
import tensorflow as tf
import pickle
from models import get_base_graph

from oracle import TimeOracle
from utils import Timer, Timeline, log_progress

__author__ = 'Sayed Hadi Hashemi'


class ExperimentResult:
    def __init__(self, **kwargs):
        self.times = []
        self.metadata = []
        self.__dict__.update(kwargs)

    def save(self, filename):
        with open(filename, "wb") as fp:
            pickle.dump(self, fp)

    def save_time_oracle(self, filename):
        oracle = TimeOracle()
        for m in self.metadata:
            oracle.update(m)
        oracle.save(filename)


class Experiment:
    def __init__(self, master, workers, base_model, ordering_algorithm, batch_size):
        self._master = master
        self._workers = workers
        self._model = base_model
        self._batch_size = batch_size
        self._ordering_algorithm = ordering_algorithm
        self._train = []
        self._loss = []
        self.get_model()

    def _get_scope(self):
        return "{}-{}".format(self._model, self._ordering_algorithm)

    def get_model(self):
        tf.reset_default_graph()
        worker_devices = [
            "/job:worker/task:{worker}".format(worker=w)
            for w in range(self._workers)
        ]
        self._train = []
        self._loss = []
        scope = self._get_scope()
        first = True
        for worker_device in worker_devices:
            with tf.variable_scope("", reuse=not first):
                first = False
                with tf.device(tf.train.replica_device_setter(worker_device=worker_device, ps_tasks=1)):
                    loss_ = get_base_graph(self._model, self._batch_size, scope)
                    self._loss.append(loss_)
                    opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
                    train_ = opt.minimize(loss_)
                    self._train.append(train_)

    def run(self, steps, stages=("fw", "train")):
        ret = []
        for stage, target in [("fw", self._loss), ("train", self._train)]:
            if stage not in stages:
                continue
            result = ExperimentResult(workers=self._workers, base_model=self._model, batch_size=self._batch_size,
                                      ordering_algorithm=self._ordering_algorithm, stage=stage, steps=steps)
            with tf.train.MonitoredTrainingSession(master=self._master) as sess:
                # Warm up run
                sess.run(target)
                for _ in log_progress(range(steps)):
                    with Timer() as timer:
                        with Timeline() as timeline:
                            sess.run(target, **timeline.kwargs())
                    result.times.append(timer.elapsed())
                    result.metadata.append(timeline.run_metadata)
            ret.append(result)
        return ret
