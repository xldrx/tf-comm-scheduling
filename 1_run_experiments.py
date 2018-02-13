#! /usr/bin/env python -u
# coding=utf-8
import argparse
import json
import os
from exps import Experiment

__author__ = 'xl'


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
    parser.add_argument("-r", "--repeat", help="Number of repeats per experiment", type=int, default=100)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    master = args.master
    workers = args.workers
    try_per_step = args.repeat

    # Load Batch Sizes
    batch_size_filename = "batch_sizes-{}.json".format(workers)
    base_models = (
        "inception_v3",
        "resnet_152",
        "vgg16",
        "alexnet",
        "seq-32",
        "par-32",
    )

    batch_size = {model: 10 for model in base_models}
    batch_size.update(load_json(batch_size_filename, {}))

    for model in base_models:
        for algorithm in ["none", "TAO", "TIO"]:
            print("//{}-{}".format(model, algorithm))
            results = Experiment(master, workers, model, algorithm, batch_size[model]).run(try_per_step)
            for stage, result in zip(["fw", "train"], results):
                filename = "{model}-{algorithm}-{stage}-{workers}.pickle".format(
                    model=model, algorithm=algorithm, stage=stage, workers=workers)
                result.save(filename)
