import pickle
import os
import numpy as np

from collections import namedtuple
from training_utils import TFSample
from training_models import CTCSpeechModel


dummy_model = CTCSpeechModel(n_speakers=96)

with open("../data/train_data.pkl", "rb") as pfile:
    train_data = pickle.load(pfile)

with open("../data/test_data.pkl", "rb") as pfile:
    test_data = pickle.load(pfile)

with open("./keyword_data.pkl", "rb") as pfile:
    mturk = pickle.load(pfile)


benchmark_train = dummy_model.create_nengo_data(mturk["train"], n_steps=1, itemize=True)
benchmark_test = dummy_model.create_nengo_data(mturk["test"], n_steps=1, itemize=True)


def check_consistency(benchmark_formatted, training_formatted):
    """Checks that data formatted for training vs benchmarks is identical"""
    for item_a, item_b in zip(benchmark_formatted, training_formatted):
        assert np.allclose(item_a[0], np.squeeze(item_b.arrays["inp"]))
        assert item_a[1] == item_b.text


check_consistency(train_data, benchmark_train)
check_consistency(test_data, benchmark_test)

print("Datasets contain identical info")
