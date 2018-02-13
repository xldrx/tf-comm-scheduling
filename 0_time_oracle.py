#! /usr/bin/env python -u
# coding=utf-8
import json
import os
from collections import OrderedDict
from exps import Experiment
from models import get_base_graph
from oracle import TimeOracle
from results import ResultAnalyser
from wizard import TAO, TIO
import tensorflow as tf
import argparse

__author__ = 'Sayed Hadi Hashemi'


def load_json(filename, default_value):
    if os.path.exists(filename):
        with open(filename, "r") as fp:
            return json.load(fp)
    return default_value


def save_json(filename, data):
    with open(filename, "w") as fp:
        return json.dump(data, fp)


def priority_print(priority_dict):
    ret = "std::unordered_map<std::string, int> rpc_list = \n{\n"
    first = True
    for name, priority in priority_dict.items():
        if first:
            first = False
        else:
            ret += ",\n\n"
        ret += "// {}\n".format(name)
        ret += ",\n".join(
            ['{"%s", %s}' % (row[1].op.name[:-5], row[0]) for row in sorted(priority, key=lambda x: x[0])])
    ret += "\n\n};"
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("master", help="Master uri e.g grpc://1.2.3.4:2222")
    parser.add_argument("workers", help="Number of workers", type=int)
    parser.add_argument("-r", "--repeat", help="Number of repeats per experiment", type=int, default=10)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    master = args.master
    workers = args.workers
    try_per_step = args.repeat

    batch_size_filename = "batch_sizes-{}.json".format(workers)
    base_models = (
        "inception_v3",
        "resnet_152",
        "vgg16",
        "alexnet",
        "seq-32",
        "par-32",
    )

    # Find Batch Size with S>90%
    batch_size = {model: 10 for model in base_models}
    batch_size.update(load_json(batch_size_filename, {}))

    for model in base_models:
        attempts = 10
        min_size = 1
        max_step = 100

        print("//{}".format(model))
        for i in range(attempts):
            print("Attempt {}:\tbatch_size: {}\t".format(i, batch_size[model]), sep="")
            result = Experiment(master, workers, model, "none", batch_size[model]).run(try_per_step, ["fw"])[0]
            a = ResultAnalyser(result).get_a()
            if 1.1 >= a >= 0.9:
                print("a={}\t(Final)".format(a))
                break
            else:
                multiplier = min(max_step, a)
                batch_size[model] = max(min_size, int(batch_size[model] * multiplier))
                print("a={}\tchange to={}".format(a, batch_size[model]))
        save_json(batch_size_filename, batch_size)

    # Estimate Time Oracle
    for model in base_models:
        print("//{}".format(model))
        result = Experiment(master, workers, model, "none", batch_size[model]).run(try_per_step, ["fw"])[0]
        oracle = TimeOracle(scope="{}-{}".format(model, "none"))
        for m in result.metadata:
            oracle.update(m)
        oracle.save("time-oracle-{}.json".format(model))

    # Estimate Time Oracle
    priorities_dict = OrderedDict()
    for model in base_models:
        print("//{}".format(model))
        try:
            tf.reset_default_graph()
            scope = "{}-{}".format(model, "TAO")
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, worker_device="/job:worker/task:0")):
                loss = get_base_graph(model, batch_size[model], scope=scope)
            oracle = TimeOracle.load("time-oracle-{}.json".format(model), scope)
            if oracle:
                tao = TAO(loss, oracle)
                priorities = tao.get_priorities()
                priorities_dict[scope] = priorities
        finally:
            pass
        try:
            tf.reset_default_graph()
            scope = "{}-{}".format(model, "TIO")
            with tf.device(tf.train.replica_device_setter(ps_tasks=1, worker_device="/job:worker/task:0")):
                loss = get_base_graph(model, batch_size[model], scope=scope)
            tio = TIO(loss)
            priorities = tio.get_priorities()
            priorities_dict[scope] = priorities
        finally:
            pass

    with open("rpc_orders.h", "w") as fp:
        fp.write(priority_print(priorities_dict))

    print('Put `rpc_orders.h` in "tensorflow/core/distributed_runtime/rpc/" and recompile TF.')
