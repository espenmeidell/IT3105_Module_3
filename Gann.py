from typing import List, Tuple, Union
from enum import Enum, auto
import numpy as np
import tensorflow as tf
from functools import partial
from CaseManager import CaseManager


class ActivationFunction(Enum):
    SIGMOID = partial(tf.nn.sigmoid)
    SOFTMAX = partial(tf.nn.softmax)
    RELU = partial(tf.nn.relu)
    TANH = partial(tf.nn.tanh)

    def __call__(self, *args):
        self.value(*args)


class VariableType(Enum):
    IN = "in"
    OUT = "out"
    WEIGHTS = "weights"
    BIASES = "biases"

class SpecType(Enum):
    AVG = auto()
    MAX = auto()
    MIN = auto()
    HIST = auto()


class Gann:
    def __init__(self,
                 dimensions: List[int],
                 case_manager: CaseManager,
                 learning_rate: float = 0.1,
                 minibatch_size: int = 10,
                 show_interval: int = 0,
                 validation_interval: int = 0,
                 initial_weight_range: Tuple[float, float] = (-0.1, 0.1),
                 output_function: ActivationFunction = ActivationFunction.SIGMOID,
                 hidden_function: ActivationFunction = ActivationFunction.RELU):
        self.layer_dimensions = dimensions
        self.case_manager = case_manager
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.show_interval = show_interval
        self.validation_interval = validation_interval
        self.initial_weight_range = initial_weight_range
        self.output_function = output_function
        self.hidden_function = hidden_function

        self.input = None
        self.output = None
        self.target = None
        self.layers: List[GannLayer] = []

    def build_neural_net(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float64,
                                    shape=(None, self.layer_dimensions[0]),
                                    name="Input")
        input_variables = self.input
        input_size = self.layer_dimensions[0]
        layer = None
        for i, output_size in enumerate(self.layer_dimensions[1:]):
            layer = GannLayer(i,
                              input_variables,
                              input_size,
                              output_size,
                              self.hidden_function,
                              self.initial_weight_range)
            input_variables = layer.output
            input_size = layer.output_size
            self.layers.append(layer)
        self.output = self.output_function(layer.output)
        self.target = tf.placeholder(tf.float64,
                                     shape=(None, layer.output_size),
                                     name="Target")
        # TODO: configure learning



class GannLayer:
    def __init__(self,
                 index: int,
                 input_variables: tf.placeholder,
                 input_size: int,
                 output_size: int,
                 activation_function: ActivationFunction,
                 initial_weight_range: Tuple[float, float] = (-0.1, 0.1)):
        self.name = "Module-" + str(index)
        self.input = input_variables
        self.input_size = input_size    # number of neurons feeding into this module
        self.output_size = output_size  # number of neurons in the module

        lower_init_weight = initial_weight_range[0]
        upper_init_weight = initial_weight_range[1]

        self.weights = tf.Variable(np.random.uniform(lower_init_weight,
                                                     upper_init_weight,
                                                     size=(self.input_size, self.output_size)),
                                   name=self.name + "-weight",
                                   trainable=True)

        self.biases = tf.Variable(np.random.uniform(lower_init_weight,
                                                    upper_init_weight,
                                                    size=self.output_size),
                                  name=self.name + "-bias",
                                  trainable=True)

        self.output = activation_function(tf.matmul(self.input, self.weights) + self.biases,
                                          name=self.name + "-out")

    def get_variable(self, type: VariableType) -> tf.Variable:
        if type == VariableType.IN:
            return self.input
        elif type == VariableType.OUT:
            return self.output
        elif type == VariableType.WEIGHTS:
            return self.weights
        else:
            return self.biases

    def generate_probe(self, type: VariableType, specs: List[SpecType]):
        variable = self.get_variable(type)
        base = self.name + "_" + type.value
        with tf.name_scope("probe_"):
            if SpecType.AVG in specs:
                tf.summary.scalar(base + "/avg/", tf.reduce_mean(variable))
            if SpecType.MAX in specs:
                tf.summary.scalar(base + "/max/", tf.reduce_max(variable))
            if SpecType.MIN in specs:
                tf.summary.scalar(base + "/min/", tf.reduce_min(variable))
            if SpecType.HIST in  specs:
                tf.summary.histogram(base + "/hist/", variable)

