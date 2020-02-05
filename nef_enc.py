import pickle
import nengo

import numpy as np

from utils import merge, allowed_text, id_to_char, create_stream
from nengo.solvers import NoSolver


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


with nengo.Network() as net:
    net.config[nengo.Connection].synapse = None

    inp = nengo.Node(size_in=inp_dim)

    layer_0 = nengo.Ensemble(n_neurons=n_neurons, dimensions=n_neurons,
                             gain=np.ones(n_neurons),
                             encoders=np.eye(n_neurons),
                             normalize_encoders=False,
                             bias=b_layer0, neuron_type=neuron_type)

    layer_1 = nengo.Ensemble(n_neurons=n_neurons, dimensions=n_neurons,
                             gain=np.ones(n_neurons),
                             encoders=np.eye(n_neurons),
                             normalize_encoders=False,
                             bias=b_layer1, neuron_type=neuron_type)

    out_bias = nengo.Node(1)

    out = nengo.Node(size_in=out_dim)

    print(W_output.T.shape)
    nengo.Connection(inp, layer_0, transform=W_layer0.T)
    nengo.Connection(layer_0, layer_1, solver=NoSolver(np.eye(n_neurons)),
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


for features, text in test_data:
    inputs = np.squeeze(features)
    n_steps = inputs.shape[0]

    feed = create_stream(inputs)
    net.inp.output = feed

    with nengo.Simulator(net) as sim:
        sim.run_steps(n_steps)

    results = sim.data[probe]
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

print("Summary")
print("=======")
print("True positive rate:\t%.3f" % (stats["tp"] / stats["aloha"]))
print("False negative rate:\t%.3f" % (stats["fn"] / stats["aloha"]))
print()
print("True negative rate:\t%.3f" % (stats["tn"] / stats["not-aloha"]))
print("False positive rate:\t%.3f" % (stats["fp"] / stats["not-aloha"]))
