import tensorflow as tf
import numpy as np
from typing import List, Tuple
from enum import Enum
import math
from CaseManager import CaseManager
import examples.tflowtools as tft


class ActivationFunction(Enum):
    SIGMOID = tf.nn.sigmoid
    SOFTMAX = tf.nn.softmax
    RELU = tf.nn.relu
    TANH = tf.nn.tanh

    def __call__(self, *args):
        self.value(*args)


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
        self.output = function(tf.matmul(input, self.weights) + self.biases,
                               name=self.name + "_out")


class NeuralNet:
    def __init__(self,
                 input_size: int,
                 layer_sizes: List[int],
                 layer_functions: List[ActivationFunction],
                 case_manager: CaseManager,
                 learning_rate: float,
                 init_weight_range: Tuple[float, float],
                 init_bias_range: Tuple[float, float]):

        assert len(layer_sizes) == len(layer_functions)

        self.session = None
        self.layers = []
        self.input_layer = None
        self.input_size = input_size
        self.output_layer = None
        self.target = None
        self.case_manager = case_manager
        self.error_history = []
        self.validation_history = []
        self.global_training_step = 0

        self._build_network(layer_sizes, layer_functions, init_weight_range, init_bias_range)

        # TODO: Generalize
        # self.error = tf.reduce_mean(tf.square(self.target - self.output_layer), name="MSE")
        self.error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.target, logits=self.output_layer))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Backprop")
        correct_predictions = tf.equal(tf.argmax(self.target, 1), tf.argmax(self.output_layer, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

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
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    def train(self, epochs: int, minibatch_size: int):
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
            acc = acc / n_minibatches
            if i % 100 == 0:
                print("\n[Training] Error at step %d is %f" % (i, error))
                print("[Training] Accuracy at step %d is %.2f%%" % (i, acc*100))



def autoencoder():
    ann = NeuralNet(input_size=16,
                    layer_sizes=[4, 16],
                    layer_functions=[ActivationFunction.RELU, ActivationFunction.SOFTMAX],
                    case_manager=CaseManager((lambda : tft.gen_all_one_hot_cases(2**4)), 0.1, 0.1),
                    learning_rate=0.3,
                    init_weight_range=(-0.1, 0.1),
                    init_bias_range=(-0.1, 0.1))
    ann.train(5000, 10)


autoencoder()
