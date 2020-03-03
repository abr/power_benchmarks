import pickle
import string

import ignite
from nni.compression.torch import QAT_Quantizer, NaiveQuantizer, BNNQuantizer
import numpy as np
import torch

from power_benchmarks.training.training_utils import merge, allowed_text


n_features = 26
n_frames = 15
hidden_layers = [256, 256]

char_list = string.ascii_lowercase + "' -"
char_to_id = {c: i for i, c in enumerate(char_list)}
id_to_char = {i: c for i, c in enumerate(char_list)}
n_chars = len(char_list)

torch.set_default_dtype(torch.float32)


class KWSpotter(torch.nn.Module):
    def __init__(self):
        super().__init__()

        prev_n = n_features * n_frames
        for i, n in enumerate(hidden_layers):
            # assign to self so that it is tracked properly by pytorch
            setattr(self, "layer_%d" % i, torch.nn.Linear(prev_n, n))
            setattr(self, "relu_%d" % i, torch.nn.ReLU())
            prev_n = n

        self.output = torch.nn.Linear(prev_n, n_chars)

    def forward(self, x):
        for i in range(len(hidden_layers)):
            x = getattr(self, "layer_%d" % i)(x)
            x = getattr(self, "relu_%d" % i)(x)
        return self.output(x)


model = KWSpotter()


print("built model")
# print(model.count_params(), model.count_params() / 173341)

# load the audio files collected from turkers
with open("keyword_data.pkl", "rb") as pfile:
    dataset = pickle.load(pfile)

# load itemized train and test data for evaluation
with open("../../data/test_data.pkl", "rb") as pfile:
    test_data = pickle.load(pfile)

with open("../../data/train_data.pkl", "rb") as pfile:
    train_data = pickle.load(pfile)

with open("../../data/ctc_data.pkl", "rb") as pfile:
    ff_data = pickle.load(pfile)

n_speakers = len(dataset["speakers"])
print("Speakers: %d" % n_speakers)
print("Testing Items: %d" % len(dataset["test"]))
print("Training Items: %d" % len(dataset["train"]))

# apply a fixed permutation to randomize the train/validation split
permutation = np.random.RandomState(0).permutation(ff_data["inp"].shape[0])

print("loaded data")

# run training
n_epochs = 20
minibatch_size = 64
validation_split = 0.2
val_split = int(len(permutation) * (1 - validation_split))
learning_rate = 6e-4
clipnorm = 10.0

x_data = torch.from_numpy(ff_data["inp"][permutation, 0])
y_data = torch.from_numpy(np.argmax(ff_data["out"][permutation, 0], axis=-1))
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[:val_split], y_data[:val_split]),
    batch_size=minibatch_size,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_data[val_split:], y_data[val_split:]),
    batch_size=minibatch_size,
)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")


def update(engine, batch):
    model.train()
    optimizer.zero_grad()
    x, y = ignite.engine._prepare_batch(batch)
    y_pred = model(x)
    loss = loss_fn(y_pred, y.long())
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipnorm)
    optimizer.step()
    return loss.item()


trainer = ignite.engine.Engine(update)

evaluator = ignite.engine.create_supervised_evaluator(
    model,
    metrics={
        "accuracy": ignite.metrics.Accuracy(),
        "loss": ignite.metrics.Loss(loss_fn),
    },
)


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    print(
        "Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}".format(
            trainer.state.epoch, metrics["accuracy"], metrics["loss"]
        )
    )


@trainer.on(ignite.engine.Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(val_loader)
    metrics = evaluator.state.metrics
    print(
        "Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}".format(
            trainer.state.epoch, metrics["accuracy"], metrics["loss"]
        )
    )


# trainer.run(train_loader, max_epochs=n_epochs)
# torch.save(model.state_dict(), "../../data/kw_weights")

checkpoint = torch.load("../../data/kw_weights")
model.load_state_dict(checkpoint)


def compute_stats(model, data, id_to_char):
    """Compute True/False Pos/Neg stats for Tensorflow keyword model"""
    stats = {"fp": 0, "tp": 0, "fn": 0, "tn": 0, "aloha": 0, "not-aloha": 0}

    for features, text in data:
        inputs = np.squeeze(features)

        with torch.no_grad():
            outputs = model(torch.from_numpy(inputs)).numpy()

        ids = np.argmax(outputs, axis=1)
        chars = [id_to_char[i] for i in ids]
        predicted_chars = merge(merge("".join(chars)))

        if text == "aloha":
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


print("Training Data Statistics:")
compute_stats(model, train_data, id_to_char)
print()
print("Testing Data Statistics:")
compute_stats(model, test_data, id_to_char)

# quantizer = QAT_Quantizer(model)
# quantizer = NaiveQuantizer(
#     model,
#     [{"quant_types": ["weight"], "quant_bits": {"weight": 8}, "op_types": ["default"]}],
# )
quantizer = QAT_Quantizer(
    model,
    [
        {
            "quant_types": ["weight"],
            "quant_bits": {"weight": 8},
            "op_types": ["default"],
        },
        {"quant_types": ["output"], "quant_bits": 10, "op_types": ["ReLU"]},
    ],
)
# quantizer = BNNQuantizer(
#     model,
#     [
#         {"quant_bits": 1, "quant_types": ["weight"], "op_types": ["default"],},
#         {"quant_bits": 1, "quant_types": ["output"], "op_types": ["ReLU"],},
#     ],
# )
quantizer.compress()

trainer.run(train_loader, max_epochs=5)
torch.save(model.state_dict(), "../../data/kw_weights_quantized")


print("Training Data Statistics:")
compute_stats(model, train_data, id_to_char)
print()
print("Testing Data Statistics:")
compute_stats(model, test_data, id_to_char)

# for m in model.modules():
#     if not hasattr(m, "name"):
#         continue
#     print(m.name)
#     print(quantizer.layer_scale[m.name])
#     print(m.module.weight / quantizer.layer_scale[m.name])
