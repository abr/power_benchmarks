import pickle
import string

import numpy as np
import tensorflow as tf

from training_models import CTCSpeechModel
from training_utils import compute_stats


n_layers = 2
n_per_layer = 256
n_features = 26
n_frames = 15

char_list = string.ascii_lowercase + "' -"
char_to_id = {c: i for i, c in enumerate(char_list)}
id_to_char = {i: c for i, c in enumerate(char_list)}
n_chars = len(char_list)

# set up ff model
inputs = x = tf.keras.Input(shape=(n_features * n_frames,))
for i in range(n_layers):
    x = tf.keras.layers.Dense(units=n_per_layer, activation=tf.nn.relu)(x)
outputs = tf.keras.layers.Dense(units=n_chars)(x)

model = tf.keras.Model(inputs, outputs)
print("built model")
print(model.inputs)
print(model.outputs)

# load the audio files collected from turkers
with open("./keyword_data.pkl", "rb") as pfile:
    dataset = pickle.load(pfile)

# load itemized train and test data for evaluation
with open("../data/test_data.pkl", "rb") as pfile:
    test_data = pickle.load(pfile)

with open("../data/train_data.pkl", "rb") as pfile:
    train_data = pickle.load(pfile)

n_speakers = len(dataset["speakers"])
print("Speakers: %d" % n_speakers)
print("Testing Items: %d" % len(dataset["test"]))
print("Training Items: %d" % len(dataset["train"]))

# load a tensorflow model that aligns audio windows with specific chars
ctc_model = CTCSpeechModel(n_speakers=n_speakers)
ctc_model.load("./checkpoints/tf_ctc_model")

# convert data to (n_items, n_steps, n_features) format for batch training
ff_data = ctc_model.create_nengo_data(dataset["train"], n_steps=1)

# apply a fixed permutation to randomize the train/validation split
permutation = np.random.RandomState(0).permutation(ff_data["inp"].shape[0])

print("loaded data")

# run training
model.compile(
    optimizer=tf.optimizers.RMSprop(5e-4, clipnorm=5.0),
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
)
model.fit(
    x=ff_data["inp"][permutation, 0],
    y=ff_data["out"][permutation, 0],
    validation_split=0.2,
    batch_size=64,
    epochs=20,
    callbacks=[
        # tf.keras.callbacks.LearningRateScheduler(
        #     tf.optimizers.schedules.ExponentialDecay(
        #         5e-4, decay_steps=8000, decay_rate=0.7, staircase=False
        #     )
        # ),
        tf.keras.callbacks.ModelCheckpoint(
            "checkpoints/keras_ff_model.tf",
            save_best_only=True,
            save_weights_only=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        tf.keras.callbacks.EarlyStopping(patience=10),
    ],
)

model.load_weights("checkpoints/keras_ff_model.tf")

print("Training Data Statistics:")
compute_stats(model, train_data, id_to_char)
print()
print("Testing Data Statistics:")
compute_stats(model, test_data, id_to_char)
