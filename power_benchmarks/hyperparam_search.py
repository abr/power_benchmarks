import pickle
import string

import nni
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

n_features = 26
n_frames = 15
char_list = string.ascii_lowercase + "' -"
n_chars = len(char_list)

with open("../data/ctc_data.pkl", "rb") as pfile:
    ff_data = pickle.load(pfile)

# apply a fixed permutation to randomize the train/validation split
permutation = np.random.RandomState(0).permutation(ff_data["inp"].shape[0])
x_data = ff_data["inp"][permutation, 0]
y_data = ff_data["out"][permutation, 0]

params = nni.get_next_parameter()
# params = {
#     "layer_0": {"_name": "Dense", "n_neurons": 256.0},
#     "layer_1": {"_name": "None"},
#     "layer_2": {"_name": "Dense", "n_neurons": 64.0},
#     "layer_3": {"_name": "Dense", "n_neurons": 128.0},
#     "layer_4": {"_name": "None"},
#     "optimizer": "rmsprop",
#     "minibatch_size": 64.0,
#     "learning_rate": 2.150807845419494e-05,
#     "grad_norm_clip": "None",
# }
print("params")
print(params)

inputs = x = tf.keras.Input(shape=(n_features * n_frames,))
for i in range(5):
    if params["layer_%d" % i]["_name"] != "None":
        x = tf.keras.layers.Dense(
            units=int(params["layer_%d" % i]["n_neurons"]), activation=tf.nn.relu
        )(x)
outputs = tf.keras.layers.Dense(units=n_chars)(x)

model = tf.keras.Model(inputs, outputs)

sparsity = 1 - model.count_params() / 173341  # based on # params in original network
print("n_params:", model.count_params())
print("sparsity:", sparsity)


class SendMetrics(tf.keras.callbacks.Callback):
    """
    Keras callback to send metrics to NNI framework
    """

    def on_epoch_end(self, epoch, logs=None):
        nni.report_intermediate_result(sparsity + logs["val_accuracy"])


opt_kwargs = {"learning_rate": params["learning_rate"]}
if params["grad_norm_clip"] != "None":
    opt_kwargs["clipnorm"] = params["grad_norm_clip"]
if params["optimizer"] == "adam":
    optimizer = tf.optimizers.Adam(**opt_kwargs)
elif params["optimizer"] == "rmsprop":
    optimizer = tf.optimizers.RMSprop(**opt_kwargs)

model.compile(
    optimizer=optimizer,
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
with tf.device("/cpu:0"):
    history = model.fit(
        x=x_data,
        y=y_data,
        validation_split=0.2,
        batch_size=int(params["minibatch_size"]),
        epochs=20,
        callbacks=[SendMetrics()],
        verbose=2,
    )

nni.report_final_result(sparsity + history.history["val_accuracy"][-1])
