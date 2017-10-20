# Project: IT_3105_Module_3
# Created: 19.10.17 13:42
import matplotlib.pyplot as plt


def plot_training_error(error_history, validation_history):
    x, y = zip(*error_history)
    plt.plot(x, y, label="Training")
    plt.ylabel("Error")
    plt.xlabel("Epoch")

    x, y = zip(*validation_history)
    plt.plot(x, y, label="Validation")
    plt.legend(loc='upper right')
    plt.title("Training and validation error")
    plt.show()