#! /usr/bin/env python -u
# coding=utf-8

__author__ = 'Sayed Hadi Hashemi'
import tensorflow as tf
import tensorflow.contrib.slim.nets as nets


class ToyModel:
    def __init__(self, parameter_size, layers, batch_size=1, scope=None):
        self.parameter_size = parameter_size
        self.layers = layers
        self.batch_size = batch_size
        self.dtype = tf.float32
        self.var_size = max(1, self.parameter_size // self.dtype.size)
        self.carry_on_size = 1000
        self.scope = scope

    def _get_layer(self, inputs, n_vars, name):
        ops = []
        with tf.name_scope("{}".format(name)):
            for i in range(n_vars):
                with tf.name_scope("op-{}".format(i)):
                    var = tf.get_variable("{}-{}".format(name, i), [self.var_size])
                    slice = tf.slice(var, [0], [self.carry_on_size], name="slice")
                    mul = tf.multiply(inputs, slice, name="mul")
                    ops.append(mul)
        output = tf.add_n(ops, name="add")
        return output

    def __call__(self, *args, **kwargs):
        with tf.variable_scope(self.scope, "Toy"):
            with tf.name_scope(self.scope, "Toy"):
                inputs = tf.ones([self.batch_size, self.carry_on_size], name="Input")
                # labels = tf.ones([self.batch_size, self.carry_on_size], name="Labels")
                model = inputs
                for i, layer in enumerate(self.layers):
                    name = "Layer-{}".format(i)
                    with tf.variable_scope(name):
                        model = self._get_layer(model, layer, name)
                return model, inputs


def get_base_graph(net, batch_size=16, scope=None):
    labels = tf.random_uniform([batch_size, 1000], name="Labels")

    if net == "inception_v3":
        inputs = tf.random_uniform([batch_size, 299, 299, 3], name="Inputs")
        model, _ = nets.inception.inception_v3(inputs, scope=scope)
    elif net == "vgg16":
        inputs = tf.random_uniform([batch_size, 224, 224, 3], name="Inputs")
        model, _ = nets.vgg.vgg_16(inputs, scope=scope)
    elif net == "resnet_152":
        inputs = tf.random_uniform([batch_size, 299, 299, 3], name="Inputs")
        model, _ = nets.resnet_v1.resnet_v1_152(inputs, 1000, scope=scope)
        model = tf.squeeze(model)
    elif net == "alexnet":
        inputs = tf.random_uniform([batch_size, 224, 224, 3], name="Inputs")
        model, _ = nets.alexnet.alexnet_v2(inputs, scope=scope)
    elif net == "par-32" or net == "seq-32":
        layer_layout = [1 for _ in range(32)] if net == "seq-32" else [32]
        model, inputs = ToyModel(2 * 2 ** 20, layer_layout, batch_size=batch_size, scope=scope)()
    else:
        return None
    with tf.name_scope(scope, "loss"):
        loss = tf.losses.mean_squared_error(model, labels, scope="loss")
    return loss
