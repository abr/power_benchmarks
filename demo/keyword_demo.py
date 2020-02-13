"""
Based on Pete's keyword spotter demo:
https://github.com/nengo/nengo-examples/tree/speech_models/speech
"""
import pickle
import numpy as np

import nengo
from nengo.solvers import NoSolver

from nengo_fpga.networks import FpgaPesEnsembleNetwork as KeywordNet

from utils import allowed_text, id_to_char, create_stream


# Load the parameters for previously trained keyword spotter
with open("../data/trained_weights_180.pkl", "rb") as pfile:
    weights = pickle.load(pfile)

# Load test data
with open("../data/test_data.pkl", "rb") as pfile:
    test_data = pickle.load(pfile)

# Params
n_neurons = 180
inp_dim = 390
out_dim = 29
neuron_type = nengo.RectifiedLinear()

n_samples = 100  # How many input samples to prepare
input_pad = 750  # How many timesteps to pad between input samples

# Trained weights
W_layer0 = weights["char_layer_0/weights"]
b_layer0 = weights["char_layer_0/biases"]

W_layer1 = weights["char_layer_1/weights"]
b_layer1 = weights["char_layer_1/biases"]

W_output = weights["char_output/weights"]
b_output = weights["char_output/biases"]

board = "pynq"  # FPGA name in the nengo-fpga config


def default_value(t, x):
    """Node function to predict blank character by default"""
    y = x + 100  # Better visualize outputs by boosting them to +ve vals
    y[-1] += 1  # Bias null char so input padding doesn't predict chars
    return y


# Keyword spotter network
with nengo.Network(seed=1) as model:
    model.config[nengo.Connection].synapse = None

    inp = nengo.Node(size_in=inp_dim, label="Input")

    # Network that may be put on FPGA
    layer_0 = KeywordNet(
        board,
        n_neurons=n_neurons,
        dimensions=n_neurons,
        learning_rate=0,
        label="Layer 0",
    )
    layer_0.ensemble.neuron_type = neuron_type
    layer_0.ensemble.bias = b_layer0
    layer_0.ensemble.encoders = np.eye(n_neurons)
    layer_0.ensemble.normalize_encoders = False
    layer_0.ensemble.gain = np.ones(n_neurons)
    layer_0.connection.solver = NoSolver(np.eye(n_neurons))
    layer_0.connection.synapse = None

    layer_1 = nengo.Ensemble(
        n_neurons=n_neurons,
        dimensions=n_neurons,
        gain=np.ones(n_neurons),
        encoders=np.eye(n_neurons),
        normalize_encoders=False,
        bias=b_layer1,
        neuron_type=neuron_type,
        label="Layer 1",
    )

    out_bias = nengo.Node(1, label="Output Bias")

    out = nengo.Node(size_in=out_dim, label="Raw Output")

    nengo.Connection(inp, layer_0.input, transform=W_layer0.T)
    nengo.Connection(
        layer_0.output,
        layer_1,
        solver=NoSolver(np.eye(n_neurons)),
        transform=W_layer1.T,
    )
    nengo.Connection(
        layer_1, out, solver=NoSolver(np.eye(n_neurons)), transform=W_output.T
    )
    nengo.Connection(out_bias, out, transform=np.expand_dims(b_output, axis=1))

    model.inp = inp

    # Filter and bias towards predicting blank symbol by default
    filtered_output = nengo.Node(
        default_value, size_in=29, size_out=29, label="Filtered Output"
    )

    nengo.Connection(out, filtered_output)


def prep_input(net, offset=1000, samples=50):
    """Prepare input samples with padding between for GUI"""
    offset = offset  # Pad zeros between samples

    all_inputs = np.zeros((offset, inp_dim))
    all_correct = []

    for features, text in test_data[:samples]:
        all_inputs = np.vstack(
            (
                all_inputs,
                np.zeros((offset, inp_dim)),
                np.squeeze(features),
                np.zeros((offset, inp_dim)),
            )
        )

        all_correct.append(text)

    feed = create_stream(all_inputs)
    model.inp.output = feed
    return offset, all_correct


offset, correct_text = prep_input(model, offset=input_pad, samples=n_samples)
dt = 0.001

# Slightly modified version of Terry's code for HTML visualization in GUI
with model:
    sample = [0]
    winsize = 1  # Single sample char, no smear
    display_count = [0]
    window_index = [0]
    window = np.zeros(winsize, dtype=int)

    def result_func(t, x):
        """Node function for displaying predicted text sequence"""

        # Reset prediction graphic
        if t / dt < offset and sample[0] > 0:
            sample[0] = 0
            result_func._nengo_html_ = (
                "<strong>Sample " + str(sample[0]) + "</strong> <hr/>"
            )

        # We aren't actually using this since we have sample data.
        # This was used to help stabilize audio inputs with a window.
        window[window_index[0]] = np.argmax(x)
        window_index[0] = (window_index[0] + 1) % winsize

        # Grab character prediction
        char = id_to_char[np.argmax(np.bincount(window))]

        # Clear prediction for new sample
        if round(t / dt) % (2 * offset) == 0:
            result_func._nengo_html_ = (
                "<strong>Sample " + str(sample[0]) + "<hr/>" +
                "Correct text: </strong>" + str(correct_text[sample[0]]) +
                "<br/><strong>Prediction: </strong> "
            )
            sample[0] += 1

        # Append new char if it's not null or repeated
        if char != "-":
            if (
                len(result_func._nengo_html_) == 0
                or result_func._nengo_html_[-1] != char
            ):
                result_func._nengo_html_ += char

        return None

    def decision_func(t, x):
        """Node function for displaying acceptance decision"""
        decision_func._nengo_html_ = "<strong>Decision:</strong> "
        phrase = result_func._nengo_html_.split()[-1]

        if phrase in allowed_text and display_count[0] < 1000:
            decision_func._nengo_html_ += '\n <font color="green"> Accepted!</font>'
            display_count[0] += 1
        else:
            decision_func._nengo_html_ += '\n <font color="red"> Not accepted!</font>'
            display_count[0] = 0

        return None

    result = nengo.Node(result_func, size_in=29, label="Characters")
    nengo.Connection(filtered_output, result)

    accept = nengo.Node(decision_func, size_in=29, label="Decision")
    nengo.Connection(filtered_output, accept)
