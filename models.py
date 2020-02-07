import string
import warnings
import numpy as np
import tensorflow as tf

# attempt to import movidius V1 api, won't work in conda environment
try:
    from mvnc import mvncapi as mvnc

    mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

    devices = mvnc.EnumerateDevices()

    if len(devices) == 0:
        print("No devices found")
        quit()

    device = mvnc.Device(devices[0])
    device.OpenDevice()

except ImportError:
    warnings.warn("NCS API not installed", UserWarning)

# attempt to import movidius V2 api, might work in conda environment
try:
    from openvino.inference_engine import IENetwork, IEPlugin

    model_bin = "./checkpoints/movidius.bin"
    model_xml = "./checkpoints/movidius.xml"

    plugin = IEPlugin(device="MYRIAD")
    network = IENetwork(model=model_xml, weights=model_bin)
    exec_net = plugin.load(network=network)

    input_blob = next(iter(network.inputs))
    output_blob = next(iter(network.outputs))

except ImportError:
    warnings.warn("NCS OpenVino API not installed", UserWarning)

# attempt to import edgetpu api, only works when ssh'd into coral dev board
try:
    from edgetpu.basic.basic_engine import BasicEngine

    model_path = "./checkpoints/movidius_edgetpu.tflite"
    tpu_engine = BasicEngine(model_path)

except ImportError:
    warnings.warn("EdgeTPU API not installed", UserWarning)


class BaseModel(object):
    """Base class with utilities used by all benchmarked speech models"""

    def __init__(self):

        char_list = string.ascii_lowercase + "' -"
        # for mapping between int IDs and characters
        self.char_to_id = {c: i for i, c in enumerate(char_list)}
        self.id_to_char = {i: c for i, c in enumerate(char_list)}
        self.n_chars = len(char_list)

    @staticmethod
    def merge(chars):
        """Merge repeated characters and strip blank CTC symbol"""
        acc = ["-"]
        for c in chars:
            if c != acc[-1]:
                acc.append(c)

        acc = [c for c in acc if c != "-"]
        return "".join(acc)

    def predict_text(self, inputs):
        """Dummy predictor getting baseline consumption values"""
        return None


class TensorflowModel(BaseModel):
    """An inference-only version of speech model for doing power consumption
    benchmarks on different kinds of hardware (Movidius, Jetson, CPU, GPU).

    Parameters:
    -----------

    n_inputs : int
        The dimensionality of the input to the model.
    n_layers : int
        The number of feedforward layers in the model.
    n_per_layer : int
        The dimensionality of each hidden layer in the model.
    """

    def __init__(self, n_inputs, n_layers=2, n_per_layer=256):

        self.n_inputs = n_inputs
        self.n_layers = n_layers
        self.n_per_layer = n_per_layer

        self.graph = tf.Graph()
        self.built = False

        # set this for initializing randomly parameterized inference models
        self.initializer = tf.random_uniform_initializer(-0.05, 0.05)

        super().__init__()

    def start_session(self):
        """Start a session instance for doing inference"""
        if self.built:
            with self.graph.as_default():
                self.sess = tf.Session(config=self.config)
                self.sess.run(tf.global_variables_initializer())
                self.saver = tf.train.Saver()
        else:
            raise RuntimeError("No graph exists to start a session with!")

    def build_branch(self, inputs, branch_scope):
        """Build a single branch from placeholder to output"""

        with tf.variable_scope(branch_scope):
            # create seperate variable scope for each layer
            scopes = ["char_layer_" + str(n) for n in range(self.n_layers)]
            x = inputs

            for i, scope in enumerate(scopes):
                size_in = self.n_inputs if i < 1 else self.n_per_layer
                size_out = self.n_per_layer

                with tf.variable_scope(scope):
                    w = tf.get_variable(
                        "weights",
                        shape=[size_in, size_out],
                        initializer=self.initializer,
                    )
                    b = tf.get_variable(
                        "biases", shape=[size_out], initializer=self.initializer
                    )

                    x = tf.nn.relu(tf.nn.xw_plus_b(x, w, b))

            # create variable scope for output layer
            with tf.variable_scope("char_output"):
                w = tf.get_variable(
                    "weights",
                    shape=[size_in, self.n_chars],
                    initializer=self.initializer,
                )
                b = tf.get_variable(
                    "biases", shape=[self.n_chars], initializer=self.initializer
                )

                outputs = tf.nn.xw_plus_b(x, w, b, name="outputs")

        return outputs

    def build(self, with_gpu=False, n_copies=1):
        """Build inference-only graph for benchmarking power consumption"""
        with self.graph.as_default():

            self.inputs = tf.placeholder(tf.float32, [1, self.n_inputs], name="inputs")

            self.copy_scopes = ["copy_" + str(c) for c in range(n_copies)]
            copy_output = []

            # make n_copies of the inference graph to feed from placeholder
            # (this is to load-test movidius, which doesn't allow batching)
            for scope in self.copy_scopes:
                output = self.build_branch(self.inputs, scope)
                copy_output.append(output)

            # sum outputs or take first output (copies just for load testing)
            # (we don't use this kind of copying in the paper, since we can't
            # replicate the required architecture on Loihi)
            if n_copies > 1:
                # add outputs to ensure all copies are executed
                self.outputs = tf.add_n(copy_output)
            else:
                self.outputs = copy_output[0]

            # set gpu config for starting session using proper hardware
            if with_gpu:
                config = tf.ConfigProto(log_device_placement=True)
                config.gpu_options.per_process_gpu_memory_fraction = 0.1
                self.config = config
                print("Set memory allocation limit")
            else:
                self.config = tf.ConfigProto(
                    log_device_placement=True, device_count={"GPU": 0}
                )

            # flag that the model has been built, so session can be made
            self.built = True

    def set_weights(self, weight_dict):
        """Assign previously trained parameters to the model variables"""
        with self.graph.as_default():
            # need to repeat weight setting for all copies of graph
            for var in tf.trainable_variables():
                var_name = "/".join(var.op.name.split("/")[1:])
                val = weight_dict[var_name]
                self.sess.run(var.assign(val))

            print("Number of var assignments: %d" % len(tf.trainable_variables()))

    def set_tensorboard_summary(self, logdir):
        """Write a tensorboard summary to file for inspecting the graph"""
        writer = tf.summary.FileWriter(logdir, graph=self.graph)
        writer.flush()
        print("Wrote graph definition to %s!" % logdir)

    def predict_text(self, inputs):
        """Feed data through the inference graph to predict text"""
        with self.graph.as_default():
            feed_dict = {self.inputs: inputs}
            outputs = self.sess.run(self.outputs, feed_dict=feed_dict)

            ids = np.argmax(outputs, axis=1)
            text = "".join(self.id_to_char[i] for i in ids)

        return text

    def save(self, checkpoint):
        """Save checkpoint files for inference model"""
        if self.sess is None:
            raise RuntimeError("No inf session object exists to save!")
        else:
            with self.graph.as_default():
                self.saver.save(self.sess, checkpoint)


class ScaledModel(TensorflowModel):
    """A model that scales the same way as scaling occurs on Loihi, for
    benchmarking differences in power consumption with changes in compute
    load under constant I/O"""

    def __init__(self, n_inputs, n_copies, n_layers, n_per_layer=256):

        super().__init__(n_inputs)

        self.n_inputs = n_inputs
        self.n_copies = n_copies
        self.n_layers = n_layers
        self.n_per_layer = n_per_layer

        self.graph = tf.Graph()
        self.built = False

    def build_layer(self, inputs, scope, from_input=False):
        """Build a single branch from placeholder to output"""
        # create seperate variable scope for each layer
        size_in = self.n_inputs if from_input else self.n_per_layer
        size_out = self.n_per_layer

        with tf.variable_scope(scope):
            w = tf.get_variable(
                "weights", shape=[size_in, size_out], initializer=self.initializer
            )

            b = tf.get_variable(
                "biases", shape=[size_out], initializer=tf.zeros_initializer()
            )

            outputs = tf.nn.relu(tf.nn.xw_plus_b(inputs, w, b))

        return outputs

    def build(self, with_gpu=False):
        """Build with internal scaling of copies, layer depth"""
        with self.graph.as_default():

            self.inputs = tf.placeholder(
                tf.float32, [None, self.n_inputs], name="inputs"
            )

            inp_layer = self.build_layer(
                self.inputs, scope="inp_layer", from_input=True
            )

            copy_scopes = ["copy_" + str(c) for c in range(self.n_copies)]
            layer_scopes = ["layer_" + str(n) for n in range(self.n_layers)]
            copy_outputs = []

            # build out over layers and copies
            for copy_scope in copy_scopes:
                current = inp_layer

                for layer_scope in layer_scopes:
                    scope = copy_scope + "/" + layer_scope
                    copy_output = self.build_layer(current, scope)
                    current = copy_output

                copy_outputs.append(copy_output)

            # project down to single output layer
            with tf.variable_scope("out_layer"):
                nx = self.n_copies if self.n_copies != 0 else 1

                w = tf.get_variable(
                    "weights",
                    shape=[self.n_per_layer * nx, self.n_per_layer],
                    initializer=self.initializer,
                )

                b = tf.get_variable(
                    "biases",
                    shape=[self.n_per_layer],
                    initializer=tf.zeros_initializer(),
                )

                # handles special case with no internal scaling
                # (equivalent to original spotter architecture)
                if self.n_copies == 0 and self.n_layers == 0:
                    activities = inp_layer
                else:
                    activities = tf.concat(copy_outputs, axis=1)

                out_layer = tf.nn.relu(tf.nn.xw_plus_b(activities, w, b))

            # create variable scope for output layer
            with tf.variable_scope("char_output"):
                w = tf.get_variable(
                    "weights",
                    shape=[self.n_per_layer, self.n_chars],
                    initializer=self.initializer,
                )

                b = tf.get_variable(
                    "biases", shape=[self.n_chars], initializer=tf.zeros_initializer()
                )

                self.outputs = tf.nn.xw_plus_b(out_layer, w, b, name="outputs")

            # set gpu config for starting session using proper hardware
            if with_gpu:
                config = tf.ConfigProto(log_device_placement=True)
                config.gpu_options.per_process_gpu_memory_fraction = 0.1
                self.config = config
                print("Set memory allocation limit")
            else:
                self.config = tf.ConfigProto(
                    log_device_placement=True, device_count={"GPU": 0}
                )

            # flag that the model has been built, so session can be made
            self.built = True


class MovidiusModel(BaseModel):
    """An inference-only version of speech model running on Movidius NCS"""

    def load_graph(self, filename):
        """Load a previosly compiled graph to run on the NCS"""
        with open(filename, mode="rb") as graph_file:
            graph = graph_file.read()

        self.model = device.AllocateGraph(graph)

    def predict_text(self, features):
        """Predict a single character from a feature input window"""
        self.model.LoadTensor(features.astype(np.float16), "user object")
        outputs, userobj = self.model.GetResult()
        idx = np.argmax(outputs)

        return self.id_to_char[idx]

    def close_graph(self):
        """Shut everything down on the NCS"""
        self.model.DeallocateGraph()
        device.CloseDevice()


class MovidiusModelV2(BaseModel):
    """An inference-only version of speech model running on Movidius NCS 2"""

    def predict_text(self, features):
        """Predict a single character from a feature input window"""
        result = exec_net.infer(inputs={input_blob: features})
        outputs = result[output_blob]
        idx = np.argmax(outputs)

        return self.id_to_char[idx]

    def close_graph(self):
        """Shut everything down on the NCS 2"""
        pass  # dummy method, avoids change to experiment script


class TPUModel(BaseModel):
    """An inference-only version of speech model running on Coral Edge TPU"""

    def predict_text(self, features):
        """Predict a single character from a feature input window"""
        _, res = tpu_engine.RunInference(np.squeeze(features).astype(np.uint8))
        idx = np.argmax(res)

        return self.id_to_char[idx]
