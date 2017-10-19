from typing import List, Tuple, Callable, Any, Union, Dict
from enum import Enum, auto
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as PLT
from CaseManager import CaseManager
from examples import tflowtools as TFT
import math
from DataReaders import read_numeric_file_with_class_in_final_column


class ActivationFunction(Enum):
    SIGMOID = tf.nn.sigmoid
    SOFTMAX = tf.nn.softmax
    RELU = tf.nn.relu
    TANH = tf.nn.tanh

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


class Layer:
    def __init__(self,
                 index: int,
                 input_variables: tf.placeholder,
                 input_size: int,
                 output_size: int,
                 activation_function: ActivationFunction,
                 initial_weight_range: Tuple[float, float] = (-0.1, 0.1)):
        self.name = "Layer-" + str(index)
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

    def get_variable(self, v_type: VariableType) -> tf.Variable:
        if v_type == VariableType.IN:
            return self.input
        elif v_type == VariableType.OUT:
            return self.output
        elif v_type == VariableType.WEIGHTS:
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





class Gann:
    def __init__(self,
                 dimensions: List[int],
                 case_manager: CaseManager,
                 error_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
                 learning_rate: float = 0.1,
                 minibatch_size: int = 10,
                 show_interval: int = 0,
                 validation_interval: int = 0,
                 initial_weight_range: Tuple[float, float] = (-0.1, 0.1),
                 output_function: ActivationFunction = ActivationFunction.SIGMOID,
                 hidden_function: ActivationFunction = ActivationFunction.RELU):
        self.layer_dimensions = dimensions
        self.case_manager = case_manager
        self.error_function = error_function
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.show_interval = show_interval
        self.validation_interval = validation_interval
        self.validation_history = []
        self.initial_weight_range = initial_weight_range
        self.output_function = output_function
        self.hidden_function = hidden_function

        self.current_session = None

        self.input = None
        self.output = None
        self.target = None
        self.layers: List[Layer] = []

        self.error = None
        self.trainer = None

        self.grabvariables = []
        self.grabvariable_figures = []
        self.probes = None

        self.error_history = []
        self.global_training_step = 0

        self.state_saver = None
        self.state_save_path = None

        self.build_neural_net()
        self.configure_learning()

    def build_neural_net(self):
        tf.reset_default_graph()
        self.input = tf.placeholder(tf.float64,
                                    shape=(None, self.layer_dimensions[0]),
                                    name="Input")
        input_variables = self.input
        input_size = self.layer_dimensions[0]
        layer = None
        for i, output_size in enumerate(self.layer_dimensions[1:]):
            layer = Layer(i,
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

    def configure_learning(self):
        self.error = self.error_function(self.target, self.output)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = optimizer.minimize(self.error, name="Backprop")

    def generate_probe(self, layer_index: int, v_type: VariableType, specs: Tuple):
        self.layers[layer_index].generate_probe(v_type, specs)

    def add_grabvariable(self, layer_index: int, v_type: VariableType = VariableType.WEIGHTS):
        self.grabvariables.append(self.layers[layer_index].get_variable(v_type))
        self.grabvariable_figures.append(PLT.figure())

    def roundup_probes(self):
        self.probes = tf.summary.merge_all()

    def do_training(self,
                    session: tf.Session,
                    cases: List[Any],
                    epochs: int=100,
                    continued: bool=False):
        for i in range(epochs):
            error = 0
            step = self.global_training_step + i
            grabvariables = [self.error] + self.grabvariables
            n_cases = len(cases)
            n_batches = math.ceil(n_cases / self.minibatch_size)
            for c_start in range(0, n_cases, self.minibatch_size):
                c_end = min(n_cases, c_start + self.minibatch_size)
                minibatch = cases[c_start:c_end]
                inputs = [c[0] for c in minibatch]
                targets = [c[1] for c in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                _, grabbed_values, _ = self.run_one_step([self.trainer],
                                                         grabvariables,
                                                         self.probes,
                                                         session=session,
                                                         feed_dict=feeder,
                                                         step=step,
                                                         show_interval=self.show_interval)
                error += grabbed_values[0]
            self.error_history.append((step, error/n_batches))
            self.consider_validation_testing(step, session)
        self.global_training_step += epochs
        TFT.plot_training_history(self.error_history,
                                  self.validation_history,
                                  xtitle="Epoch",
                                  ytitle="Title",
                                  title="",
                                  fig=not continued)

    def consider_validation_testing(self, epoch: int, session: tf.Session):
        if self.validation_interval and (epoch % self.validation_interval == 0):
            cases = self.case_manager.get_validation_cases()
            if len(cases) > 0:
                error = self.do_testing(session, cases, message="Validation Testing")
                self.validation_history.append((epoch, error))

    def do_testing(self, session: tf.Session, cases: List[Any], message="Testing"):
        inputs = [c[0] for c in cases]
        targets = [c[1] for c in cases]
        feeder = {self.input: inputs, self.target: targets}
        error, grabbed_values, _ = self.run_one_step(self.error,
                                                     self.grabvariables,
                                                     self.probes,
                                                     session=session,
                                                     feed_dict=feeder,
                                                     show_interval=0)
        print('%s Set Error = %f ' % (message, error))
        return error

    def training_session(self,
                         epochs: int,
                         session: Union[None, tf.Session],
                         dir:str = "probeview",
                         continued: bool = False):
        self.roundup_probes()
        session = session if session else TFT.gen_initialized_session(dir=dir)
        self.current_session = session
        self.do_training(session, self.case_manager.get_training_cases(), epochs, continued=continued)

    def testing_session(self, session: tf.Session):
        cases = self.case_manager.get_testing_cases()
        if len(cases) > 0:
            self.do_testing(session, cases, message="Final Testing")

    def test_on_training(self, session: tf.Session):
        self.do_testing(session, self.case_manager.get_training_cases(), message="Total Training")

    def run_one_step(self,
                     operators: Any,
                     grabbed_variables: List[tf.Tensor],
                     probed_variables: List[tf.Tensor],
                     dir: str = "probeview",
                     session: Union[None, tf.Session] = None,
                     feed_dict: Union[None, Dict[Any, Any]] = None,
                     step: int = 1,
                     show_interval: int = 1):
        sess = session if session else TFT.gen_initialized_session(dir=dir)
        if probed_variables is not None:
            results = sess.run([operators, grabbed_variables, probed_variables],
                               feed_dict=feed_dict)
            sess.probe_stream.add_summary(results[2], global_step=step)
        else:
            results = sess.run([operators, grabbed_variables], feed_dict=feed_dict)
        if show_interval and (step % show_interval == 0):
            self.display_grabbed_variables(results[1], grabbed_variables, step)
        return results[0], results[1], sess

    def display_grabbed_variables(self,
                                  grabbed_values: List[Any],
                                  grabbed_variables: List[Any],
                                  step: int = 1
                                  ):
        names = [x.name for x in grabbed_variables]
        message = "Grabbed Variables at Step " + str(step)
        print("\n" + message, end="\n")
        fig_index = 0
        for i, value in enumerate(grabbed_values):
            if names:
                print("   " + names[i] + " = ", end="\n")
            # Use Hinton plot if the value is a matrix
            if type(value) == np.ndarray and len(value.shape) > 1:
                TFT.hinton_plot(value,
                                fig=self.grabvariable_figures[fig_index],
                                title=names[i] + " at step " + str(step))
                fig_index += 1
            else:
                print(value, end="\n\n")

    def save_session_parameters(self,
                                spath: str = "netsaver/my_saved_session",
                                session: Union[None, tf.Session] = None,
                                step: int = 0):
        session = session if session else self.current_session
        state_variables = []
        for l in self.layers:
            variables = [l.get_variable(VariableType.WEIGHTS), l.get_variable(VariableType.BIASES)]
            state_variables = state_variables + variables
        self.state_saver = tf.train.Saver(state_variables)
        self.state_save_path = self.state_saver.save(session, spath, global_step=step)

    def restore_session_parameters(self,
                                   path: Union[str, None] = None,
                                   session: Union[tf.Session, None] = None):
        path = path if path else self.state_save_path
        session = session if session else self.current_session
        self.state_saver.restore(session, path)

    def close_current_session(self):
        self.save_session_parameters(session=self.current_session)
        TFT.close_session(self.current_session, view=True)

    def reopen_current_session(self):
        self.current_session = TFT.copy_session(self.current_session)
        self.current_session.run(tf.global_variables_initializer())
        self.restore_session_parameters()

    def run(self, epochs: int = 100, session: Union[None, tf.Session] = None, continued: bool = False):
        PLT.ion()
        self.training_session(epochs, session, continued=continued)
        self.test_on_training(session=self.current_session)
        self.testing_session(session=self.current_session)
        self.close_current_session()
        PLT.ioff()

    def runmore(self, epochs=100):
        self.reopen_current_session()
        self.run(epochs, session=self.current_session, continued=True)




def mse(target: tf.Tensor, output: tf.Tensor) -> tf.Tensor:
    return tf.reduce_mean(tf.square(target - output), name='MSE')


def autoex(epochs=1000, nbits=4, lrate=0.03, showint=100, mbs=None, vfrac=0.1, tfrac=0.1, vint=100, sm=False):
    size = 2 ** nbits
    mbs = mbs if mbs else size
    case_generator = (lambda: TFT.gen_all_one_hot_cases(2 ** nbits))
    case_manager = CaseManager(case_generator, vfrac, tfrac)
    ann = Gann([size, nbits, size],
               case_manager,
               mse,
               lrate,
               mbs,
               showint,
               vint,
               output_function=ActivationFunction.SOFTMAX)
    ann.generate_probe(0, VariableType.WEIGHTS, (SpecType.HIST, SpecType.AVG))
    ann.generate_probe(1, VariableType.OUT, (SpecType.AVG, SpecType.MAX))
    ann.add_grabvariable(0)
    ann.run(epochs)
    correct = tf.equal(tf.argmax(ann.target, 1), tf.argmax(ann.output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, "float"))
    print("Accuracy")
    ann.reopen_current_session()
    inputs = [c[0] for c in case_manager.get_training_cases()]
    targets = [c[1] for c in case_manager.get_training_cases()]
    print(ann.current_session.run(accuracy, feed_dict={ann.input: inputs, ann.target: targets}))


# def yeast():
#     case_generator = lambda: read_numeric_file_with_class_in_final_column("data/yeast.txt",
#                                                                           normalize_parameters=True)
#     case_manager = CaseManager(case_generator, 0.2, 0.1)
#     ann = Gann([8, 5, 5, 5, 11],
#                case_manager,
#                mse,
#                0.1,
#                minibatch_size=10,
#                show_interval=100,
#                validation_interval=100,
#                initial_weight_range=(-0.1, 0.1),
#                output_function=ActivationFunction.RELU)
#     ann.generate_probe(3, VariableType.WEIGHTS, (SpecType.HIST, SpecType.AVG))
#     ann.generate_probe(1, VariableType.OUT, (SpecType.AVG, SpecType.MAX))
#     ann.add_grabvariable(0)
#     ann.run(600)
#     parameters = case_manager.get_testing_cases()[0][0]
#     target = case_manager.get_testing_cases()[0][1]
#     print("--------")
#     print(parameters)
#     print(target)
#     print("Prediction:")
#     feeder = {ann.input: [parameters], ann.target: [target]}
#     ann.reopen_current_session()
#     x = ann.current_session.run([ann.output], feed_dict=feeder)
#     print(x)
#     ann.close_current_session()


autoex()