import tensorflow as tf
import numpy as np
from typing import List, Tuple, Callable, Union
from enum import Enum
import math
import random
from CaseManager import CaseManager
import examples.tflowtools as tft
from termcolor import colored
from Plotting import plot_training_error
from pprint import pprint
from DataReaders import read_numeric_file_with_class_in_final_column
random.seed(123)
tf.set_random_seed(123)


class ActivationFunction(Enum):
    SIGMOID = tf.nn.sigmoid
    SOFTMAX = tf.nn.softmax
    RELU = tf.nn.relu
    TANH = tf.nn.tanh
    ELU = tf.nn.elu
    SOFTPLUS = tf.nn.softplus
    LRELU = "lrelu"
    LINEAR = "linear"


    def __call__(self, *args):
        self.value(*args)


def mse(output: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(target - output), name="MSE")


def cross_entropy(output: tf.Tensor, target: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=output), name="CrossEnt")


class NNLayer:
    def __init__(self,
                 index: int,
                 function: ActivationFunction,
                 input: tf.Tensor,
                 input_size: int,
                 size: int,
                 init_weight_range: Tuple[float, float],
                 init_bias_range: Tuple[float, float]):
        self.name = "Layer_" + str(index)
        self.input = input
        self.weights = tf.Variable(np.random.uniform(init_weight_range[0],
                                                     init_weight_range[1],
                                                     size=(input_size, size)),
                                   name=self.name + "_wgt",
                                   trainable=True)
        self.biases = tf.Variable(np.random.uniform(init_bias_range[0],
                                                    init_bias_range[1],
                                                    size=size),
                                  name=self.name + "_bias",
                                  trainable=True)
        if function == ActivationFunction.LRELU:
            tmp = tf.matmul(input, self.weights) + self.biases
            self.output = tf.maximum(tmp, 0.01*tmp, name=self.name + "_out")
        elif function == ActivationFunction.LINEAR:
            self.output = tf.matmul(input, self.weights) + self.biases
        else:
            self.output = function(tf.matmul(input, self.weights) + self.biases,
                                   name=self.name + "_out")


class NeuralNet:
    def __init__(self,
                 input_size: int,
                 layer_sizes: List[int],
                 layer_functions: List[ActivationFunction],
                 case_manager: CaseManager,
                 learning_rate: float,
                 error_function: Callable = mse,
                 init_weight_range: Tuple[float, float]=(-0.1, 0.1),
                 init_bias_range: Tuple[float, float]=(-0.1, 0.1)):

        assert len(layer_sizes) == len(layer_functions)

        self.session = None
        self.layers = []
        self.input_layer = None
        self.input_size = input_size
        self.output_layer = None
        self.target = None
        self.case_manager = case_manager
        self.training_error_history = []
        self.validation_error_history = []
        self.global_training_step = 0

        self._build_network(layer_sizes, layer_functions, init_weight_range, init_bias_range)

        self.error = error_function(self.output_layer, self.target)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Backprop")
        correct_predictions = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.output_layer, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Variables to monitor
        self.monitoring = []
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def _build_network(self,
                       layer_sizes: List[int],
                       layer_functions: List[ActivationFunction],
                       init_weight_range: Tuple[float, float],
                       init_bias_range: Tuple[float, float]):
        tf.reset_default_graph()
        input_size = self.input_size
        self.input_layer = tf.placeholder(tf.float64, (None, input_size), "Input")
        input = self.input_layer
        for i in range(len(layer_sizes)):
            layer = NNLayer(i,
                            layer_functions[i],
                            input,
                            input_size,
                            layer_sizes[i],
                            init_weight_range,
                            init_bias_range)
            input = layer.output
            input_size = layer_sizes[i]
            self.output_layer = layer.output
            self.layers.append(layer)
        self.target = tf.placeholder(tf.float64, (None, input_size), "Target")

    def train(self, epochs: int, minibatch_size: int, validation_interval: int=100):
        print("\nStarting training. Epochs: %d, Minibatch size: %d" % (epochs, minibatch_size))
        cases = self.case_manager.get_training_cases()
        for i in range(epochs):
            error = 0.0
            acc = 0.0
            n_cases = len(cases)
            n_minibatches = math.ceil(n_cases/minibatch_size)
            for c_start in range(0, n_cases, minibatch_size):
                batch = cases[c_start: min(n_cases, c_start + minibatch_size)]
                inputs = [c[0] for c in batch]
                targets = [c[1] for c in batch]
                feeder = {self.input_layer: inputs, self.target: targets}
                _, e, a = self.session.run([self.trainer, self.error, self.accuracy], feed_dict=feeder)
                error += e
                acc += a
            error = error / n_minibatches
            self.training_error_history.append((i, error))
            acc = acc / n_minibatches
            if validation_interval > 0 and i % validation_interval == 0:
                self.test(True, i)

            if validation_interval > 0 and i % validation_interval == 0:
                print(colored("\n[Training] Error at step %d is %f" % (i, error), "green"))
                print(colored("[Training] Accuracy at step %d is %.2f%%" % (i, acc*100), "green"))
            if validation_interval == 0 and i % 100 == 0:
                print(colored("\n[Training] Error at step %d is %f" % (i, error), "green"))
                print(colored("[Training] Accuracy at step %d is %.2f%%" % (i, acc*100), "green"))
        # Extra validation testing when done for graph
        if validation_interval > 0:
            self.test(True, epochs)

    def test(self, validation: bool=False, epoch: int=0):
        print("\nStarting validation") if validation else print("\nStarting testing")
        cases = self.case_manager.get_validation_cases() if validation else self.case_manager.get_testing_cases()
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input_layer: inputs, self.target: targets}
        e, a = self.session.run([self.error, self.accuracy], feed_dict=feeder)
        color = "red"
        if validation:
            self.validation_error_history.append((epoch, e))
            color = "yellow"
        print(colored("Testing error: %f" % e, color))
        print(colored("Testing accuracy: %.2f%%" % (a*100), color))

    def monitor(self,
                n_cases: int,
                input: bool=True,
                output: bool=True,
                dendrogram: List=[],
                layers: Union[None, List[List]]=None):
        cases = self.case_manager.get_training_cases()[0:n_cases]
        variables = []
        titles = []
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input_layer: inputs, self.target: targets}

        if input:
            variables.append(self.input_layer)
            titles.append("Input Layer")
        if output:
            variables.append(self.output_layer)
            titles.append("Output Layer")

        for spec in layers:
            if spec[1] == "w":
                variables.append(self.layers[spec[0]].weights)
                titles.append("Layer %d weights" % spec[0])
            else:
                variables.append(self.layers[spec[0]].biases)
                titles.append("Layer %d biases" % spec[0])

        result = self.session.run(variables, feed_dict=feeder)

        for i in range(len(result)):
            if result[i].ndim == 1:
                tft.display_matrix(np.array([result[i]]), title=titles[i])
            else:
                tft.hinton_plot(result[i], title=titles[i])

        if len(dendrogram) > 0:
            for l in dendrogram:
                i, a = self.session.run([self.input_layer, self.layers[l].input], feed_dict=feeder)
                labels = list(map(tft.bits_to_str, i.astype(int)))
                tft.dendrogram(a, labels)

