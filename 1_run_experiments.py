#! /usr/bin/env python -u
# coding=utf-8
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
    for name, priority in priority_dict.items():
        ret += "// {}\n".format(name)
        ret += ",\n".join(
            ['{"%s", %s}' % (row[1].op.name[:-5], row[0]) for row in sorted(priority, key=lambda x: x[0])])
        ret += "\n\n"
    ret += "\n};"
    return ret


if __name__ == '__main__':
    master = "192.17.176.131"
    workers = 4
    batch_size_filename = "batch_sizes-{}.json".format(workers)
    base_models = (
        "inception_v3",
        "resnet_152",
        "vgg16",
        "alexnet",
        "seq-32",
        "par-32"
    )

    # Load Batch Size with S>90%
    batch_size = {model: 10 for model in base_models}
    batch_size.update(load_json(batch_size_filename, {}))

    try_per_step = 1
    for model in base_models:
        for algorithm in ["none", "TAO", "TIO"]:
            print("//{}-{}".format(model, algorithm))
            results = Experiment(master, workers, model, algorithm, batch_size[model]).run(try_per_step)
            for stage, result in zip(["fw", "train"], results):
                filename = "{model}-{algorithm}-{stage}-{workers}.pickle".format(
                    model=model, algorithm=algorithm, stage=stage, workers=workers)
                result.save(filename)