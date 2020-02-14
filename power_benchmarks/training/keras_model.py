import pickle
import string

import numpy as np
import tensorflow as tf
from tensorflow_model_optimization.sparsity import keras as sparsity

from training_models import CTCSpeechModel
from training_utils import compute_stats


n_features = 26
n_frames = 15
hidden_layers = [32, 32, 32]

char_list = string.ascii_lowercase + "' -"
char_to_id = {c: i for i, c in enumerate(char_list)}
id_to_char = {i: c for i, c in enumerate(char_list)}
n_chars = len(char_list)

# set up ff model
inputs = x = tf.keras.Input(shape=(n_features * n_frames,))
for n in hidden_layers:
    x = tf.keras.layers.Dense(units=n, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(units=n_chars)(x)

model = tf.keras.Model(inputs, outputs)
print("built model")
print(model.inputs)
print(model.outputs)
print(model.count_params(), model.count_params() / 173341)

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

# load a tensorflow model that aligns audio windows with specific chars
# ctc_model = CTCSpeechModel(n_speakers=n_speakers)
# ctc_model.load("./checkpoints/tf_ctc_model")
#
# # convert data to (n_items, n_steps, n_features) format for batch training
# ff_data = ctc_model.create_nengo_data(dataset["train"], n_steps=1)
#
# with open("../../data/ctc_data.pkl", "wb") as pfile:
#     pickle.dump(ff_data, pfile)

# apply a fixed permutation to randomize the train/validation split
permutation = np.random.RandomState(0).permutation(ff_data["inp"].shape[0])

print("loaded data")

# run training
n_epochs = 20
minibatch_size = 64
validation_split = 0.2
opt_kwargs = dict(learning_rate=6e-4, clipnorm=10.0)

model.compile(
    optimizer=tf.optimizers.Adam(**opt_kwargs),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
with tf.device("/cpu:0"):
    model.fit(
        x=ff_data["inp"][permutation, 0],
        y=ff_data["out"][permutation, 0],
        validation_split=validation_split,
        batch_size=minibatch_size,
        epochs=n_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "../../data/checkpoints/keras_ff_model.tf",
                save_best_only=True,
                save_weights_only=True,
                monitor="val_accuracy",
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5),
        ],
        verbose=2,
    )


model.load_weights("../../data/checkpoints/keras_ff_model.tf")

print("Training Data Statistics:")
compute_stats(model, train_data, id_to_char)
print()
print("Testing Data Statistics:")
compute_stats(model, test_data, id_to_char)


# prune the model
pruning_epochs = 20

end_step = np.ceil(
    (1 - validation_split) * ff_data["inp"].shape[0] / minibatch_size
).astype(np.int32) * (pruning_epochs - 0)

pruned_model = sparsity.prune_low_magnitude(
    model,
    pruning_schedule=sparsity.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=end_step
    ),
)
pruned_model.summary()


def get_sparsity(m):
    return 1 - sum(
        np.count_nonzero(w)
        for w in tf.keras.backend.batch_get_value(m.trainable_weights)
    ) / sum(np.prod(w.shape) for w in m.trainable_weights)


print("Initial sparsity", get_sparsity(pruned_model))

pruned_model.compile(
    optimizer=tf.optimizers.Adam(**opt_kwargs),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

with tf.device("/cpu:0"):
    pruned_model.fit(
        x=ff_data["inp"][permutation, 0],
        y=ff_data["out"][permutation, 0],
        validation_split=validation_split,
        batch_size=minibatch_size,
        epochs=pruning_epochs,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                "../../data/checkpoints/keras_ff_model_pruned.tf",
                save_weights_only=True,
            ),
            sparsity.UpdatePruningStep(),
        ],
        verbose=2,
    )

print("Final sparsity", get_sparsity(pruned_model))

print("Training Data Statistics:")
compute_stats(pruned_model, train_data, id_to_char)
print()
print("Testing Data Statistics:")
compute_stats(pruned_model, test_data, id_to_char)
