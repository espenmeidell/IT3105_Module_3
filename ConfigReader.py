# Project: IT_3105_Module_3
# Created: 20.10.17 09:43
import json
from pprint import pprint
import time
from typing import Callable, Dict
from NeuralNet import ActivationFunction, cross_entropy, mse, NeuralNet
from DataReaders import parity, bit_count, wine, yeast, glass, seg_count, mnist, auto, iris
from CaseManager import CaseManager
from Plotting import plot_training_error

current_milli_time = lambda: int(round(time.time() * 1000))
start_time = current_milli_time()


def log(msg: str, level: str = "INFO"):
    print("[%s][Time: %6d ms]: %s" % (level, current_milli_time() - start_time, msg))


path = "run_configs/iris.json"

with open(path) as file:
    data = json.loads(file.read())


def string_to_func(s: str) -> ActivationFunction:
    if s == "relu":
        return ActivationFunction.RELU
    elif s == "softmax":
        return ActivationFunction.SOFTMAX
    elif s == "tanh":
        return ActivationFunction.TANH
    elif s == "sigmoid":
        return ActivationFunction.SIGMOID
    assert False, "Invalid activation function: %s" % s


def string_to_error(s: str) -> Callable:
    if s == "crossent":
        return cross_entropy
    elif s == "mse":
        return mse
    assert False, "Invalid error function: %s" % s


def string_to_reader(s: str) -> Callable:
    if s == "parity":
        return parity
    elif s == "bit_count":
        return bit_count
    elif s == "wine":
        return wine
    elif s == "yeast":
        return yeast
    elif s == "glass":
        return glass
    elif s == "seg_count":
        return seg_count
    elif s == "mnist":
        return mnist
    elif s == "auto":
        return auto
    elif s == "iris":
        return iris
    assert False, "Invalid data source: %s" % s


def data_to_neural_net(data: Dict) -> NeuralNet:
    cman = CaseManager(case_function=data["case_manager"]["reader"],
                       fraction=data["case_manager"]["fraction"],
                       validation_fraction=data["case_manager"]["validation"],
                       testing_fraction=data["case_manager"]["test"])

    return NeuralNet(input_size=data["net"]["input_size"],
                     layer_sizes=data["net"]["layer_sizes"],
                     layer_functions=data["net"]["functions"],
                     case_manager=cman,
                     learning_rate=data["net"]["learning_rate"],
                     error_function=data["net"]["error"],
                     init_weight_range=tuple(data["net"]["weight_range"]),
                     init_bias_range=tuple(data["net"]["bias_range"]))


print("\nRunning configuration: %s\n" % path)
pprint(data)


data["net"]["functions"] = list(map(string_to_func, data["net"]["functions"]))
data["net"]["error"] = string_to_error(data["net"]["error"])
data["case_manager"]["reader"] = string_to_reader(data["case_manager"]["reader"])



network = data_to_neural_net(data)

network.train(epochs=data["training"]["epochs"],
              minibatch_size=data["training"]["minibatch_size"],
              validation_interval=data["training"]["validation_interval"])

print("\nFinished training after %d seconds" % ((current_milli_time() - start_time)/1000))

network.test()

plot_training_error(network.training_error_history, network.validation_error_history)

#network.monitor()

#input()