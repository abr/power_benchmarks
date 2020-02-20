import pickle
import string

from nni.nas.pytorch import callbacks, enas
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.nas.pytorch.utils import AverageMeter
import numpy as np
import torch


# monkeypatch AverageMeter.update.
# the EnasTrainer passes reward as a pytorch tensor (which is outside our control),
# which causes this to spit out a bunch of warnings that make the log output hard
# to read.
def update(self, val, n=1):
    if not isinstance(val, float) and not isinstance(val, int):
        # CHANGE: replacing this line with the one below
        # _logger.warning("Values passed to AverageMeter must be number, not %s.", type(val))
        val = val.item()
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


AverageMeter.update = update


n_features = 26
n_frames = 15

char_list = string.ascii_lowercase + "' -"
char_to_id = {c: i for i, c in enumerate(char_list)}
id_to_char = {i: c for i, c in enumerate(char_list)}
n_chars = len(char_list)

torch.set_default_dtype(torch.float32)


class KWSpotter(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.max_n = 256

        self.hidden_layers = []
        self.skips = []
        prev_n = n_features * n_frames
        for i in range(5):
            self.hidden_layers.append(
                LayerChoice(
                    [torch.nn.Linear(prev_n, n) for n in range(32, self.max_n + 1, 32)]
                )
            )
            self.skips.append(InputChoice(n_candidates=1))

            # we don't know what the prev_n will be, so we'll always assume 256
            # (and pad it out if it isn't)
            prev_n = self.max_n

            # make sure all the layers are tracked (since they aren't directly
            # added to self)
            self.add_module("layer_%d" % i, self.hidden_layers[-1])
            self.add_module("skip_%d" % i, self.skips[-1])

        self.output = torch.nn.Linear(prev_n, n_chars)

    def forward(self, x):
        for i in range(5):
            new_x = self.hidden_layers[i](x)

            # potentially skip this layer
            new_x = self.skips[i]([new_x])
            if new_x is None:
                new_x = x

            # pad it back to 256
            x = torch.nn.functional.pad(new_x, (0, self.max_n - new_x.shape[-1]))
        return self.output(x)


model = KWSpotter()

with open("../data/ctc_data.pkl", "rb") as pfile:
    ff_data = pickle.load(pfile)

# apply a fixed permutation to randomize the train/validation split
permutation = np.random.RandomState(0).permutation(ff_data["inp"].shape[0])

# run training
n_epochs = 1000
minibatch_size = 64
validation_split = 0.2
val_split = int(len(permutation) * (1 - validation_split))
learning_rate = 6e-4
clipnorm = 10.0
device = torch.device("cpu")

x_data = torch.from_numpy(ff_data["inp"][permutation, 0])
y_data = torch.from_numpy(np.argmax(ff_data["out"][permutation, 0], axis=-1))

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")


def count_params():
    # calculates size of sampled model
    # from https://github.com/microsoft/nni/issues/1947

    x = torch.ones((1, n_features * n_frames)).to(device)
    y = model(x)
    target = torch.ones((1,)).long().to(device)
    loss = loss_fn(y, target)

    # do a backwards pass to check which parameters are actually hit (this will
    # detect the parameters used in the current sampled model)
    for v in model.parameters():
        v.grad = None
    loss.backward()
    return sum(np.prod(v.size()) for v in model.parameters() if v.grad is not None)


def sparsity_accuracy(y_pred, y_true):
    n_params = count_params()
    sparsity = 1 - n_params / 173341  # based on # params in original network
    sparse_weight = 0.5

    result = (
        torch.mean(torch.eq(torch.argmax(y_pred, dim=-1), y_true).float()).item()
        + sparse_weight * sparsity
    )

    return result


class SearchCheckpoint(callbacks.Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def on_epoch_end(self, epoch):
        torch.save(
            {
                "model": self.model.state_dict(),
                "mutator": self.mutator.state_dict(),
                "mutator_optim": self.trainer.mutator_optim.state_dict(),
            },
            self.filename,
        )
        self.trainer.export(file="%s.architecture.json" % self.filename)


trainer = enas.EnasTrainer(
    model,
    loss=loss_fn,
    metrics=lambda y_pred, y_true: {
        "sparsity_accuracy": sparsity_accuracy(y_pred, y_true)
    },
    reward_function=sparsity_accuracy,
    optimizer=optimizer,
    batch_size=minibatch_size,
    num_epochs=n_epochs,
    dataset_train=torch.utils.data.TensorDataset(
        x_data[:val_split], y_data[:val_split]
    ),
    dataset_valid=torch.utils.data.TensorDataset(
        x_data[val_split:], y_data[val_split:]
    ),
    workers=0,
    callbacks=[SearchCheckpoint("../data/checkpoints/nas_search")],
    log_frequency=100,
    device=device,
)

# resume from checkpoint
# checkpoint = torch.load("../data/checkpoints/nas_search")
# trainer.model.load_state_dict(checkpoint["model"])
# trainer.mutator.load_state_dict(checkpoint["mutator"])
# trainer.mutator_optim.load_state_dict(checkpoint["mutator_optim"])

trainer.train()
