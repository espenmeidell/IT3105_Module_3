# Project: IT_3105_Module_3
# Created: 25.10.17 10:42
from termcolor import colored
from ConfigReader import read_config, current_milli_time, start_time
from Plotting import plot_training_error


def print_blue(s: str):
    print(colored(s, "blue"))


config_list = ["parity", "bit_count", "segment_count", "wine", "yeast", "glass", "mnist", "auto", "iris"]


def main():
    print_blue("\n###################################################################")
    print_blue("######################## NEURAL NET RUNNER ########################")
    print_blue("###################################################################\n")

    print_blue("Available configurations: \n%s\n" % ", ".join(config_list))

    config_name = input(colored("Enter name of configuration: ", "blue"))

    network, data = read_config(config_name)

    n_epochs = data["training"]["epochs"]

    network.train(epochs=n_epochs,
                  minibatch_size=data["training"]["minibatch_size"],
                  validation_interval=data["training"]["validation_interval"])

    print_blue("\nFinished training after %d seconds" % ((current_milli_time() - start_time)/1000))

    if data["case_manager"]["test"]:
        network.test()

    plot_training_error(network.training_error_history, network.validation_error_history)

    if "monitoring" in data.keys():
        specs = data["monitoring"]
        network.monitor(n_cases=specs["n_cases"],
                        input=specs["input"],
                        output=specs["output"],
                        layers=specs["layers"],
                        dendrogram=specs["dendrogram"])
        input()



main()
