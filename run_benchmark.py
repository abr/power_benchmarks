import argparse
import pickle
import json
import time
import numpy as np

from datetime import datetime
from models import TensorflowModel, MovidiusModel, MovidiusModelV2, ScaledModel

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--movidius", action="store_true")
parser.add_argument("--movidius_2", action="store_true")
parser.add_argument("--mov_graph", type=str)
parser.add_argument("--tpu", action="store_true")
parser.add_argument("--tpu_graph", type=str)
parser.add_argument("--bsize", type=int)
parser.add_argument("--n_copies", type=int)
parser.add_argument("--n_layers", type=int)
parser.add_argument("--nx_neurons", type=int)
parser.add_argument("--time", type=int)
parser.add_argument("--log", type=str)
args = parser.parse_args()

# default to single copy, single batch if unspecified
args.bsize = args.bsize if args.bsize is not None else 1
args.n_copies = args.n_copies if args.n_copies is not None else 1
args.nx_neurons = args.nx_neurons if args.nx_neurons is not None else 1

# load data for piping into the keyword spotter
with open("./data/test_data.pkl", "rb") as pfile:
    data = pickle.load(pfile)

# load the parameters for previously trained keyword spotter
with open("./data/inference_weights.pkl", "rb") as pfile:
    weights = pickle.load(pfile)


# handles cpu, gpu, and jetson (which is a gpu)
if args.cpu or args.gpu:
    # build a scaled model w/ n_layers, n_copies and random weights
    if args.n_layers is not None:
        model = ScaledModel(
            n_inputs=390, n_layers=args.n_layers, n_copies=args.n_copies
        )
        model.build(with_gpu=args.gpu)
        model.start_session()

    # build the model to use 2 layers, with optional scaling of neuron count
    else:
        model = TensorflowModel(
            n_inputs=390, n_layers=2, n_per_layer=args.nx_neurons * 256
        )
        model.build(with_gpu=args.gpu, n_copies=args.n_copies)
        model.start_session()

    # load the pretrained weights if the there's no extra layers, neurons
    if args.nx_neurons == 1 and args.n_layers == None:
        model.set_weights(weights)

    # flag jetson in log since it runs with GPU command line argument
    if args.log and "jetson" not in args.log:
        hardware = "CPU" if args.cpu else "GPU"
    else:
        hardware = "JETSON"

    # write graph to tensorboard to inspect
    model.set_tensorboard_summary("./tensorboard")

# handles the case of using the movidius NCS
elif args.movidius:
    model = MovidiusModel()
    model.load_graph(args.mov_graph)

    hardware = "MOVIDIUS"

# handles the case of using the movidius NCS 2
elif args.movidius_2:
    model = MovidiusModelV2()

    hardware = "MOVIDIUS_2"

# handles the case of using the Coral TPU Board
elif args.tpu:
    model = TPUModel()

    hardware = "TPU"

else:
    raise Exception("No hardware specified to run benchmark on!")


def make_batches(data, batchsize):
    # stack feature frames for convient batch slicing
    data = [np.squeeze(sample[0]) for sample in data]
    stacked_data = np.concatenate(data, axis=0)

    # truncate to make all batches the same size
    n_batches = len(stacked_data) // batchsize
    stacked_data = stacked_data[: n_batches * batchsize]

    return np.split(stacked_data, n_batches)


batches = make_batches(data, args.bsize)
print("Number of batches: %d" % len(batches))

time.sleep(5)  # sleep to distance from setup power consumption

step_count = 0
start_time = time.time()
start_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("Loop starting...")
while True:
    # loop over batches until time limit is reached
    for batch in batches:
        text = model.predict_text(batch)
        step_count += 1

        if time.time() - start_time > args.time:
            break
    else:
        continue
    break

end_tag = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print("Elapsed time: %4f" % (time.time() - start_time))
print("Target time: %2f" % args.time)
print("Number of inferences: %d" % (step_count * args.bsize))

if args.movidius:
    model.close_graph()

# write a json summmary of this benchmarking experiment
if args.log:
    summary = {}
    summary["hardware"] = hardware
    summary["start_time"] = "_".join(start_tag.split(" "))
    summary["end_time"] = "_".join(end_tag.split(" "))
    summary["nx_neurons"] = args.nx_neurons
    summary["n_copies"] = args.n_copies
    summary["n_layers"] = args.n_layers
    summary["n_seconds"] = args.time
    summary["n_inferences"] = step_count * args.bsize
    summary["inf_per_second"] = summary["n_inferences"] / summary["n_seconds"]
    summary["batchsize"] = args.bsize
    summary["log_name"] = args.log.split("/")[-1]  # use file name, not path
    summary["status"] = "Running"

    with open(args.log, "w") as jfile:
        json.dump(summary, jfile)
