import argparse
import pickle
import json
import time
import os
import numpy as np

from datetime import datetime
from models import TensorflowModel, ScaledModel

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=str)
parser.add_argument("--n_copies", type=int)
parser.add_argument("--n_layers", type=int)
parser.add_argument("--nx_neurons", type=int)
parser.add_argument("--scaled", action="store_true")
args = parser.parse_args()

args.n_copies = args.n_copies if args.n_copies else 1
args.nx_neurons = args.nx_neurons if args.nx_neurons else 1

# handles 2 layer keyword spotter, either functional or with nx_neurons/layer
if not args.scaled:

    with open("./data/inference_weights.pkl", "rb") as pfile:
        weights = pickle.load(pfile)

    # build the model using weights from previously trained model
    model = TensorflowModel(n_inputs=390, n_layers=2, n_per_layer=args.nx_neurons * 256)
    model.build(n_copies=args.n_copies)
    model.start_session()

    if args.nx_neurons == 1:
        model.set_weights(weights)

    model.save(args.ckpt)
    model.set_tensorboard_summary("./tensorboard")

    os.system(
        "mvNCCompile %s.meta -s 12 -in inputs -on copy_0/char_output/outputs -o %s.graph"
        % (args.ckpt, args.ckpt)
    )

    print("Movidius graph compiled successfully!")

# handles case of scaling network with multiple copies/layers (see paper)
else:
    model = ScaledModel(n_inputs=390, n_copies=args.n_copies, n_layers=args.n_layers)
    model.build()
    model.start_session()
    model.save(args.ckpt)
    model.set_tensorboard_summary("./tensorboard/")

    os.system(
        "mvNCCompile %s.meta -s 12 -in inputs -on char_output/outputs -o %s.graph"
        % (args.ckpt, args.ckpt)
    )

    print("Movidius graph compiled successfully!")
