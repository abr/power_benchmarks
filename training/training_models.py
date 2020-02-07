import features
import pickle
import random
import string

import numpy as np
import tensorflow as tf

from copy import deepcopy
from training_utils import with_graph, normalize
from training_utils import NengoSample, build_arrays, convert_to_onehot


class CTCSpeechModel(object):
    """Feedforward neural network model that performs a combination of speech
    recognition and speaker identification. The model takes a sequence of audio
    features as input (from a sliding window over the raw audio waveform), and
    predicts a corresponding sequence of probability distributions over
    characters and speaker IDs. For each sequence of these distributions,
    there is a set of valid paths corresponding the target sequence of labels,
    and the objective of training (via CTC loss) is minimize the negative
    log-liklihood of all valid paths corresponding to the target sequence.
    Different paths exist due to the fact that different output sequences can
    collapse to the same labelling after merging repeated characters and
    deleting the blank CTC symbol.
    *Important*: it is possible for training to result in instances for which
    there are no valid paths through the sequence of output distributions. This
    occurs if the model mistakenly assigns zero probability to the outputs that
    are required by all valid paths at a given timestep. Because softmax func
    magnifies differences between positive and negative output logits, this can
    happen fairly easily, especially with high learning rates.
    See https://distill.pub/2017/ctc/ for helpful details and explanations.
    Parameters
    ----------
    n_speakers : int
        The number of unique speakers in the training data.
    feature : str (optional)
        The name of the function to use for computing features.
    n_per_layer : int (optional)
        The number of neurons in each network layer.
    n_layers : int (optional)
        The total number of layers in the network.
    n_shared : int (optional)
        Number of layers shared for recognition and identification objectives.
    n_features : int (optional)
        Number of filters or cepstra to use for computing auditory features.
    n_frames : int (optional)
        Number of frames (i.e. audio spans used to compute features) to include
        in each input window.
    char_list : list (optional)
        The characters to select from when predicting output labels.
    checkpoints : str
        The path to where checkpoint files will be saved.
    """

    def __init__(
        self,
        n_speakers,
        feature="mfcc",
        n_per_layer=256,
        n_layers=4,
        n_shared=0,
        n_features=26,
        n_frames=15,
        char_list=None,
        checkpoints=None,
    ):

        if not hasattr(features, feature):
            raise ValueError("Feature %r not found" % feature)

        self.checkpoints = checkpoints
        self.feature = feature
        self.n_per_layer = n_per_layer
        self.n_layers = n_layers
        self.n_shared = n_shared
        self.n_speakers = n_speakers
        self.n_features = n_features
        self.n_frames = n_frames

        self.reset()
        self.init = tf.contrib.layers.variance_scaling_initializer()

        self._feat_f = None
        self._feat_args = None

        # build a lookup tables for mapping between characters and indices
        if not char_list:
            char_list = string.ascii_lowercase + "' -"

        self.char_to_id = {c: i for i, c in enumerate(char_list)}
        self.id_to_char = {i: c for i, c in enumerate(char_list)}
        self.n_chars = len(char_list)

    @property
    def size_in(self):
        return self.n_features * self.n_frames

    def save(self, filename):
        """Save the model to a checkpoint file"""
        if self.sess is None:
            raise RuntimeError("No session object exists to save!")
        else:
            with self.graph.as_default():
                self.saver.save(self.sess, filename)

    def load(self, filename):
        """Load an existing model from a checkpoint file"""
        self.reset()
        self.build()
        self.start_session()
        with self.graph.as_default():
            self.saver.restore(self.sess, filename)

    def reset(self):
        """Reset the comp graph, remove session and saver"""
        self.graph = tf.Graph()
        self.built = False
        self.sess = None
        self.saver = None

    def check_build_status(self):
        """Ensures built graph and running session are available"""
        if not self.built:
            self.build()
            self.start_session()

    def get_features(self, audio):
        if self._feat_f is None:
            self._feat_f = getattr(features, self.feature)
            self._feat_args = (
                {"n_cepstra": self.n_features}
                if self.feature == "mfcc"
                else {"n_filters": self.n_features}
            )
        return self._feat_f(audio, **self._feat_args)

    def get_sparse_ctc_targets(self, text):
        """Convert text to tuple for sparse_tensor_array in CTC loss comp"""
        targets = np.asarray([self.char_to_id[c] for c in text])
        indices = list(zip([0] * len(targets), range(len(targets))))
        targets = (indices, targets, [1, len(targets)])

        return targets

    def ff_layer(self, inputs, size_in, size_out, scope, logits=False):
        """Build a feedforward layer that computes activations from inputs"""
        with tf.variable_scope(scope):
            w = tf.get_variable(
                "weights", shape=[size_in, size_out], initializer=self.init
            )
            b = tf.get_variable("biases", shape=[size_out], initializer=self.init)
        if logits:
            activations = tf.nn.xw_plus_b(inputs, w, b)
        else:
            activations = tf.nn.relu(tf.nn.xw_plus_b(inputs, w, b))

        return activations

    def build_ff_branch(self, x, scopes):
        """Build branch of layers specific to either id or character output"""
        for i, scope in enumerate(scopes):
            if i < 1 and self.n_shared < 1:
                size_in = self.size_in  # handles case w/ no shared layers
            else:
                size_in = self.n_per_layer

            x = self.ff_layer(x, size_in, self.n_per_layer, scope)

        return x

    def build_feed(self, audio):
        """Make feed_dict for passing audio features to model for inference"""
        features = self.get_features(audio)
        feed_dict = {self.features: features}

        return feed_dict

    @with_graph
    def start_session(self):
        """Start a session instance for doing training or prediction"""
        if self.built:
            # add this to disable GPU
            config = tf.ConfigProto(device_count={"GPU": 0})
            self.sess = tf.Session(config=config)

            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()

            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())

        else:
            raise RuntimeError("No graph exists to start a session with!")

    @with_graph
    def build(self, ctc_beam_search=False, decay_steps=8000, decay_rate=0.7):
        """Build all necessary ops into the object's tensorflow graph"""
        if self.built:
            raise RuntimeError("Graph has already been built! Please reset.")

        self.rate = tf.placeholder(tf.float32, shape=[])

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            self.rate, global_step, decay_steps, decay_rate, staircase=False
        )

        self.features = tf.placeholder(tf.float32, [None, None])
        self.speaker = tf.placeholder(tf.int32, [None])
        self.targets = tf.sparse_placeholder(tf.int32)

        n_windows = tf.shape(self.features)[0] - self.n_frames

        c_logit_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        i_logit_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        d_vec_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        window_array = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

        loop_vars = [
            0,
            n_windows,
            self.features,
            c_logit_array,
            i_logit_array,
            window_array,
            d_vec_array,
        ]

        # define loop that applies the feedforward model over a frame sequence
        def cond(t, t_stop, *args):
            # stop iterating when the full frame sequence has been encoded
            return t < t_stop

        def body(t, t_stop, features, char_logits, id_logits, windows, d_vecs):

            n_per_branch = self.n_layers - self.n_shared

            shared = ["shared_layer_" + str(n) for n in range(self.n_shared)]
            char_scopes = ["char_layer_" + str(n) for n in range(n_per_branch)]
            id_scopes = ["id_layer_" + str(n) for n in range(n_per_branch)]

            char_out_scope = "char_output"
            id_out_scope = "id_output"

            # slice window out of feature array and flatten it
            inp = self.features[t : t + self.n_frames, :]
            windows = windows.write(t, tf.reshape(inp, [1, self.size_in]))

            x = tf.reshape(inp, [1, self.size_in])

            # build and stack shared feedforward layers
            for i, scope in enumerate(shared):
                size_in = self.size_in if i < 1 else self.n_per_layer
                x = self.ff_layer(x, size_in, self.n_per_layer, scope)

            if n_per_branch > 0:
                x_char = self.build_ff_branch(x, char_scopes)
                x_id = self.build_ff_branch(x, id_scopes)
            else:
                x_char = x
                x_id = x

            # build output layers for each task
            char_out = self.ff_layer(
                x_char, self.n_per_layer, self.n_chars, char_out_scope, logits=True
            )
            id_out = self.ff_layer(
                x_id, self.n_per_layer, self.n_speakers, id_out_scope, logits=True
            )

            # accumulate logit values for each window
            char_logits = char_logits.write(t, char_out)
            id_logits = id_logits.write(t, id_out)

            # accumulate ID d-vectors
            d_vecs = d_vecs.write(t, x_id)

            return [t + 1, t_stop, features, char_logits, id_logits, windows, d_vecs]

        # note that because there are no dependencies between time steps, we
        # can run the loop iterations in parallel (doesn't make much of a diff)
        loop_output = tf.while_loop(cond, body, loop_vars, parallel_iterations=20)

        # use squeeze to create 2D instead of 3D arrays
        self.c_logits = loop_output[3].stack()  # can't squeeze b/c ctc loss
        self.i_logits = tf.squeeze(loop_output[4].stack())
        self.all_windows = tf.squeeze(loop_output[5].stack())
        self.d_vecs = tf.squeeze(loop_output[6].stack())

        n_windows = tf.expand_dims(n_windows, 0)

        char_loss = tf.nn.ctc_loss(
            self.targets,
            self.c_logits,
            n_windows,
            preprocess_collapse_repeated=False,
            ctc_merge_repeated=True,
            ignore_longer_outputs_than_inputs=True,
            time_major=True,
        )

        id_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.i_logits, labels=self.speaker
        )

        # TODO: figure out a good weighting scheme for combining losses
        self.cost = tf.reduce_sum(id_loss) + tf.reduce_sum(char_loss)

        # build the loss and an op for doing parameter updates
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)  # avoid explosions

        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self.train_step = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step
        )

        self.speaker_decode = tf.argmax(self.i_logits, axis=-1, output_type=tf.int32)
        if ctc_beam_search:
            self.char_decode, _ = tf.nn.ctc_beam_search_decoder(
                self.c_logits, n_windows
            )
        else:
            self.char_decode, _ = tf.nn.ctc_greedy_decoder(self.c_logits, n_windows)

        self.ler = tf.reduce_mean(
            tf.edit_distance(tf.cast(self.char_decode[0], tf.int32), self.targets)
        )

        self.built = True

    def train(self, dataset, rate, n_epochs=10, display=True, resume=False):
        """Train the model on a dataset of unbatched audio/text pairs"""
        if resume:
            self.load(self.checkpoints)
        else:
            self.reset()
            self.build()
            self.start_session()

        accum_loss = 0
        unshuffled_data = deepcopy(dataset)  # so we don't break order for IDs

        for epoch in range(n_epochs):
            random.shuffle(dataset)

            for item in dataset:
                features = self.get_features(item.audio)
                targets = self.get_sparse_ctc_targets(item.text)

                n_input_frames = features.shape[0] - self.n_frames
                speaker = item.speaker_id * np.ones(n_input_frames)

                feed_dict = {
                    self.rate: rate,
                    self.features: features,
                    self.targets: targets,
                    self.speaker: speaker,
                }

                cost, _ = self.sess.run(
                    [self.cost, self.train_step], feed_dict=feed_dict
                )

                # TF warnings are sufficient, don't want inf cost for epoch
                if not np.isinf(cost):
                    accum_loss += cost

            if display:
                print("Status: Epoch %d" % int(epoch + 1))
                print("Avg. loss: %2f" % (accum_loss / int(len(dataset))))
                accum_loss = 0

        self.compute_speaker_vectors(unshuffled_data)
        self.saver.save(self.sess, self.checkpoints)

    @with_graph
    def predict_chars(self, audio):
        """Predict a character transcription from an audio sample"""
        self.check_build_status()
        feed_dict = self.build_feed(audio)

        decodings = self.sess.run(self.char_decode, feed_dict=feed_dict)
        prediction = "".join([self.id_to_char[i] for i in decodings[0][1]])

        return prediction

    @with_graph
    def predict_speaker(self, audio):
        """Predict the speaker id from an audio sample"""
        self.check_build_status()
        feed_dict = self.build_feed(audio)

        decodings = self.sess.run(self.speaker_decode, feed_dict=feed_dict)
        prediction = np.argmax(np.bincount(decodings))

        return prediction

    @with_graph
    def predict_speaker_from_d_vector(self, audio):
        """Predict the speaker id from an audio sample"""
        self.check_build_status()
        feed_dict = self.build_feed(audio)

        d_vecs = self.sess.run(self.d_vecs, feed_dict=feed_dict)
        average = np.mean(d_vecs, axis=0)

        cosines = np.dot(self.speaker_array, average)
        prediction = self.speaker_mapping[np.argmax(cosines)]

        return prediction

    @with_graph
    def compute_logits(self, audio):
        """Pairs feature windows with logits for character prediction"""
        self.check_build_status()
        feed_dict = self.build_feed(audio)

        logits = self.sess.run(self.c_logits, feed_dict=feed_dict)
        windows = self.sess.run(self.all_windows, feed_dict=feed_dict)

        return windows, logits

    @with_graph
    def compute_speaker_vectors(self, dataset):
        """Compute and store average d-vector for each speaker in dataset"""
        self.check_build_status()
        self.speaker_averages = []
        self.speaker_mapping = []

        speaker_ids = set([d.speaker_id for d in dataset])

        for speaker_id in speaker_ids:
            speaker_items = [x for x in dataset if x.speaker_id == speaker_id]
            speaker_vectors = []

            for item in speaker_items:
                feed_dict = self.build_feed(item.audio)
                d_vecs = self.sess.run(self.d_vecs, feed_dict=feed_dict)
                speaker_vectors.append(normalize(np.mean(d_vecs, axis=0)))

            speaker_average = np.mean(np.vstack(speaker_vectors), axis=0)
            self.speaker_averages.append(normalize(speaker_average))
            self.speaker_mapping.append(speaker_id)

        self.speaker_array = np.vstack(self.speaker_averages)

        # check there is one average speaker vector per speaker
        assert self.speaker_array.shape[0] == len(speaker_ids)

    @with_graph
    def compute_ler(self, audio, text):
        """Compute the label error rate from an audio/text pair"""
        self.check_build_status()

        features = self.get_features(audio)
        targets = self.get_sparse_ctc_targets(text)

        feed_dict = {self.features: features, self.targets: targets}
        ler = self.sess.run(self.ler, feed_dict=feed_dict)

        return ler

    @with_graph
    def save_inference_params(self, filename):
        """Save dict mapping vars to vals for creating inference-only model"""
        variables = [v for v in tf.trainable_variables() if "char" in v.name]
        params = {v.op.name: self.sess.run(v) for v in variables}

        with open(filename, "wb") as pfile:
            pickle.dump(params, pfile, protocol=2)

    def save_quantized_model(self, filename):
        """Save protobuff file for model with quantization aware training"""
        tf.contrib.quantize.create_eval_graph(input_graph=self.graph)
        # Save the checkpoint and eval graph proto to disk for freezing
        with open(filename, "w") as f:
            f.write(str(self.graph.as_graph_def()))

    def id_error_rate(self, dataset, use_d_vectors=False):
        """Compute ID prediction error rate using supplied dataset"""
        count = 0
        for item in dataset:
            if use_d_vectors:
                predicted_id = self.predict_speaker_from_d_vector(item.audio)
            else:
                predicted_id = self.predict_speaker(item.audio)
            if predicted_id == item.speaker_id:
                count += 1

        return 100 - 100 * (count / len(dataset))

    def label_error_rate(self, dataset):
        """Compute average label error rate using supplied dataset"""
        acc = []
        for item in dataset:
            ler = self.compute_ler(item.audio, item.text)
            acc.append(ler)

        return 100 * (sum(acc) / len(acc))

    def collect_ce_data(self, dataset):
        """Group features, char onehots, id onehots as predicted by model"""
        acc = []
        for sample in dataset:
            features, logits = self.compute_logits(sample.audio)
            char_onehots = convert_to_onehot(np.squeeze(logits))
            id_onehots = np.zeros((len(features), self.n_speakers))
            id_onehots[range(len(features)), sample.speaker_id] = 1

            acc.append(dict(inp=features, out=char_onehots, ids=id_onehots))

        return acc

    def create_nengo_data(self, dataset, n_steps, itemize=False, stream=False):
        """Make Nengo DL formatted data with model predictions as CE targets"""
        if itemize:
            # create seperate items for each audio sample
            nengo_data = []
            for sample in dataset:
                ce_data = self.collect_ce_data([sample])
                arrays = build_arrays(ce_data, n_steps, stream=stream)
                data_item = NengoSample(
                    arrays, sample.text, sample.speaker_id, sample.sample_id
                )

                nengo_data.append(data_item)
        else:
            # just make one big array (for training in nengo DL)
            ce_data = self.collect_ce_data(dataset)
            nengo_data = build_arrays(ce_data, n_steps, stream=stream)

        return nengo_data


class FFSpeechModel(CTCSpeechModel):
    """Train a feedforward MLP to predict the target character distributions
    learned by a speech model trained with the CTC objective function. This
    results in model trained comparably to SNN models run on neuromorphic 
    hardware like Intel's Loihi chip. The CTC model is learning to align 
    particular audio feature windows with particular character labels, and 
    once this alignment is performed, the feedforward model learns this mapping
    directly.

    Parameters:
    ----------
    n_per_layers: int (optional)
        The dimensionality of each hidden layer in the model.
    n_layers: int (optional)
        The number of hidden layers in the model.
    checkpoints: filepath (optional)
        Name of the checkpoint file to save/load with.
    """

    def __init__(self, n_per_layer=256, n_layers=2, checkpoints=None):

        self.checkpoints = checkpoints
        self.n_per_layer = n_per_layer
        self.n_layers = n_layers
        self.n_shared = 0
        self.n_features = 26
        self.n_frames = 15

        self.reset()
        self.init = tf.contrib.layers.variance_scaling_initializer()

        char_list = string.ascii_lowercase + "' -"

        self.char_to_id = {c: i for i, c in enumerate(char_list)}
        self.id_to_char = {i: c for i, c in enumerate(char_list)}
        self.n_chars = len(char_list)

    @with_graph
    def build(self, decay_steps=8000, decay_rate=0.7, quantize=False):
        """Build all necessary ops into the object's tensorflow graph"""
        if self.built:
            raise RuntimeError("Graph has already been built! Please reset.")

        self.rate = tf.placeholder(tf.float32, shape=[])

        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(
            self.rate, global_step, decay_steps, decay_rate, staircase=False
        )

        self.inputs = tf.placeholder(tf.float32, [None, None])
        self.targets = tf.placeholder(tf.int32, [None, None])

        # build the feedforward branch of the network
        char_scopes = ["char_layer_" + str(n) for n in range(self.n_layers)]
        char_out_scope = "char_output"

        x_char = self.build_ff_branch(self.inputs, char_scopes)
        self.char_out = self.ff_layer(
            x_char, self.n_per_layer, self.n_chars, char_out_scope, logits=True
        )

        self.cost = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.targets, logits=self.char_out
            )
        )

        if quantize:
            tf.contrib.quantize.create_training_graph(input_graph=self.graph)

        # build the loss and an op for doing parameter updates
        tvars = tf.trainable_variables()
        grads = tf.gradients(self.cost, tvars)
        grads, _ = tf.clip_by_global_norm(grads, 5.0)  # avoid explosions

        optimizer = tf.train.RMSPropOptimizer(learning_rate)

        self.train_step = optimizer.apply_gradients(
            zip(grads, tvars), global_step=global_step
        )

        self.built = True

    def train(self, data, rate, bsize=64, n_epochs=10, resume=False):
        """Train the model on a dataset of unbatched audio/text pairs"""
        if resume:
            self.load(self.checkpoints)
        else:
            self.reset()
            self.build()
            self.start_session()

        accum_loss = 0

        # create batches from learned alignments
        inputs = np.squeeze(data["inp"])
        targets = np.squeeze(data["out"])

        batches = []

        for idx in range(0, len(inputs), bsize):
            batch_inputs = inputs[idx : idx + bsize]
            batch_targets = targets[idx : idx + bsize]

            batch = (batch_inputs, batch_targets)
            batches.append(batch)

        for epoch in range(n_epochs):
            random.shuffle(batches)

            for batch in batches:

                feed_dict = {
                    self.rate: rate,
                    self.inputs: batch[0],
                    self.targets: batch[1],
                }

                cost, _ = self.sess.run(
                    [self.cost, self.train_step], feed_dict=feed_dict
                )

                # TF warnings are sufficient, don't want inf cost for epoch
                if not np.isinf(cost):
                    accum_loss += cost

            print("Status: Epoch %d" % int(epoch + 1))
            print("Avg. loss: %2f" % (accum_loss / int(len(batches))))
            accum_loss = 0

        self.saver.save(self.sess, self.checkpoints)

    @with_graph
    def predict_text(self, inputs):
        """Feed data through the inference graph to predict text"""
        self.check_build_status()

        feed_dict = {self.inputs: inputs}
        outputs = self.sess.run(self.char_out, feed_dict=feed_dict)

        ids = np.argmax(outputs, axis=1)
        text = "".join(self.id_to_char[i] for i in ids)

        return text
