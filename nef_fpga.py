import pickle
import nengo

import numpy as np

from utils import merge, allowed_text, id_to_char, create_stream
from nengo.solvers import NoSolver

import nengo_fpga
from nengo_fpga.networks import FpgaPesEnsembleNetwork


# load the parameters for previously trained keyword spotter
with open('./data/trained_weights_180.pkl', 'rb') as pfile:
    weights = pickle.load(pfile)

# load test data
with open('./data/test_data.pkl', 'rb') as pfile:
    test_data = pickle.load(pfile)


n_neurons = 180

inp_dim = 390
out_dim = 29

neuron_type = nengo.RectifiedLinear()

W_layer0 = weights['char_layer_0/weights']
b_layer0 = weights['char_layer_0/biases']

W_layer1 = weights['char_layer_1/weights']
b_layer1 = weights['char_layer_1/biases']

W_output = weights['char_output/weights']
b_output = weights['char_output/biases']

# FPGA names in the nengo-fpga config
fpga_0 = 'pynq'
fpga_1 = 'pynq'

# Flag for easily switching between software and hardware implementations
# 0 = layer_0 on fpga; 1 = layer_1 on fpga; 2 = both; else all software
USE_FPGA = -1

nengo.rc.set("precision", "bits", "32")

with nengo.Network() as net:
    net.config[nengo.Connection].synapse = None

    inp = nengo.Node(size_in=inp_dim)

    if USE_FPGA in [0, 2]:
        # Create FPGA ens for layer 0
        fpga_0_ens = FpgaPesEnsembleNetwork(
            fpga_0,
            n_neurons=n_neurons,
            dimensions=n_neurons,
            learning_rate=0,
            )
        fpga_0_ens.ensemble.neuron_type = neuron_type
        fpga_0_ens.ensemble.bias = b_layer0
        fpga_0_ens.ensemble.encoders = np.eye(n_neurons)
        fpga_0_ens.ensemble.normalize_encoders = False
        fpga_0_ens.ensemble.gain = np.ones(n_neurons)
        fpga_0_ens.connection.solver = NoSolver(np.eye(n_neurons))
        fpga_0_ens.connection.transform = W_layer1.T
        fpga_0_ens.connection.synapse = None
    else:
        layer_0 = nengo.Ensemble(n_neurons=n_neurons, dimensions=n_neurons,
                                 gain=np.ones(n_neurons),
                                 encoders=np.eye(n_neurons),
                                 normalize_encoders=False,
                                 bias=b_layer0, neuron_type=neuron_type)

    if USE_FPGA in [1, 2]:
        # Create FPGA ens for layer 1
        fpga_1_ens = FpgaPesEnsembleNetwork(
            fpga_1,
            n_neurons=n_neurons,
            dimensions=n_neurons,
            learning_rate=0,
            )
        fpga_1_ens.ensemble.neuron_type = neuron_type
        fpga_1_ens.ensemble.bias = b_layer1
        fpga_1_ens.ensemble.encoders = np.eye(n_neurons)
        fpga_1_ens.ensemble.normalize_encoders = False
        fpga_1_ens.ensemble.gain = np.ones(n_neurons)
        fpga_1_ens.connection.solver = NoSolver(np.eye(n_neurons))
        # fpga_1_ens.connection.transform = W_output.T
        fpga_1_ens.connection.synapse = None
    else:
        layer_1 = nengo.Ensemble(n_neurons=n_neurons, dimensions=n_neurons,
                                 gain=np.ones(n_neurons),
                                 encoders=np.eye(n_neurons),
                                 normalize_encoders=False,
                                 bias=b_layer1, neuron_type=neuron_type)

    out_bias = nengo.Node(1)

    out = nengo.Node(size_in=out_dim)

    if USE_FPGA == 0:
        nengo.Connection(inp, fpga_0_ens.input, transform=W_layer0.T)
        nengo.Connection(fpga_0_ens.output, layer_1)
        nengo.Connection(layer_1, out,
                         solver=NoSolver(np.eye(n_neurons)),
                         transform=W_output.T)

    elif USE_FPGA == 1:
        nengo.Connection(inp, layer_0, transform=W_layer0.T)
        nengo.Connection(layer_0, fpga_1_ens.input,
                         solver=NoSolver(np.eye(n_neurons)),
                         transform=W_layer1.T)
        # FPGA didn't like the shape mismatch of the transform?
        nengo.Connection(fpga_1_ens.output, out, transform=W_output.T)

    elif USE_FPGA == 2:  # Untested so far
        nengo.Connection(inp, fpga_0_ens.input, transform=W_layer0.T)
        nengo.Connection(fpga_0_ens.output, fpga_1_ens.input)
        # FPGA didn't like the shape mismatch of the transform?
        nengo.Connection(fpga_1_ens.output, out, transform=W_output.T)

    else:
        nengo.Connection(inp, layer_0, transform=W_layer0.T)
        nengo.Connection(layer_0, layer_1,
                         solver=NoSolver(np.eye(n_neurons)),
                         transform=W_layer1.T)
        nengo.Connection(layer_1, out, solver=NoSolver(np.eye(n_neurons)),
                         transform=W_output.T)

    nengo.Connection(out_bias, out, transform=np.expand_dims(b_output, axis=1))

    probe = nengo.Probe(out)

    net.inp = inp


stats = {
    "fp": 0,
    "tp": 0,
    "fn": 0,
    "tn": 0,
    "aloha": 0,
    "not-aloha": 0
}

# Precompute inputs so we can a single simulator,
# it's slow to spin up a simulator on the FPGA
all_inputs = []
for features, text in test_data:
    if len(all_inputs) == 0:
        all_inputs = np.squeeze(features)
    else:
        all_inputs = np.vstack((all_inputs, np.squeeze(features)))

feed = create_stream(all_inputs)
net.inp.output = feed

idx = 0
with nengo_fpga.Simulator(net) as sim:
    for features, text in test_data:
        inputs = np.squeeze(features)
        n_steps = inputs.shape[0]

        sim.run_steps(n_steps)

        results = sim.data[probe][idx:]
        ids = np.argmax(results, axis=1)
        chars = [id_to_char[x] for x in ids]

        predicted_chars = merge(merge(''.join(chars)))
        print('Correct: ', text)
        print('Predicted: ', predicted_chars)
        print('')

        if text == 'aloha':
            stats["aloha"] += 1
            if predicted_chars in allowed_text:
                stats["tp"] += 1
            else:
                stats["fn"] += 1
        else:
            stats["not-aloha"] += 1
            if predicted_chars in allowed_text:
                stats["fp"] += 1
            else:
                stats["tn"] += 1

        idx += n_steps

print("Summary")
print("=======")
print("True positive rate:\t%.3f" % (stats["tp"] / stats["aloha"]))
print("False negative rate:\t%.3f" % (stats["fn"] / stats["aloha"]))
print()
print("True negative rate:\t%.3f" % (stats["tn"] / stats["not-aloha"]))
print("False positive rate:\t%.3f" % (stats["fp"] / stats["not-aloha"]))
