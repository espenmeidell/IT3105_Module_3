# Project: IT_3105_Module_3
# Created: 20.10.17 09:43
import json
from pprint import pprint
import time
from typing import Callable, Dict, Tuple
from NeuralNet import ActivationFunction, cross_entropy, mse, NeuralNet
from DataReaders import parity, bit_count, wine, yeast, glass, seg_count, mnist, auto, iris
from CaseManager import CaseManager


def current_milli_time(): return int(round(time.time() * 1000))


start_time = current_milli_time()


def string_to_func(s: str) -> ActivationFunction:
    if s == "relu":
        return ActivationFunction.RELU
    elif s == "softmax":
        return ActivationFunction.SOFTMAX
    elif s == "tanh":
        return ActivationFunction.TANH
    elif s == "sigmoid":
        return ActivationFunction.SIGMOID
    elif s == "elu":
        return ActivationFunction.ELU
    elif s == "softplus":
        return ActivationFunction.SOFTPLUS
    elif s == "lrelu":
        return ActivationFunction.LRELU
    elif s == "linear":
        return ActivationFunction.LINEAR
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


def read_config(name: str) -> Tuple[NeuralNet, Dict]:
    path = "run_configs/%s.json" % name

    with open(path) as file:
        data = json.loads(file.read())

    print("\nRunning configuration: %s\n" % path)
    pprint(data)

    data["net"]["functions"] = list(map(string_to_func, data["net"]["functions"]))
    data["net"]["error"] = string_to_error(data["net"]["error"])
    data["case_manager"]["reader"] = string_to_reader(data["case_manager"]["reader"])

    return data_to_neural_net(data), data

