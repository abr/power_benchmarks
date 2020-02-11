import argparse
import pickle

from power_benchmarks.models import (
    TensorflowModel,
    MovidiusModel,
    MovidiusModelV2,
    TPUModel,
)
from power_benchmarks.utils import compute_tf_stats

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--tpu", action="store_true")
parser.add_argument("--movidius", action="store_true")
parser.add_argument("--movidius_2", action="store_true")
parser.add_argument("--mov_graph", type=str)
args = parser.parse_args()

# load parameters and data
with open("../data/test_data.pkl", "rb") as pfile:
    test_data = pickle.load(pfile)

with open("../data/train_data.pkl", "rb") as pfile:
    train_data = pickle.load(pfile)

if args.cpu or args.gpu:
    with open("../data/inference_weights.pkl", "rb") as pfile:
        weights = pickle.load(pfile)

    # build the model using weights from previously trained model
    model = TensorflowModel(n_inputs=390, n_layers=2)
    model.build(with_gpu=args.gpu, n_copies=1)
    model.start_session()
    model.set_weights(weights)

elif args.movidius:
    model = MovidiusModel()
    model.load_graph(args.mov_graph)

elif args.movidius_2:
    model = MovidiusModelV2()

elif args.tpu:
    model = TPUModel()

else:
    raise Exception("No hardware specified to run accuracy check on!")

# print whole-word spotting accuracy
print("Training Data Statistics:")
compute_tf_stats(model, train_data)
print("")
print("Testing Data Statistics:")
compute_tf_stats(model, test_data)
