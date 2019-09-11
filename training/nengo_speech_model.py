import pickle
import nengo
import nengo_dl

from rate_model import TFSpeechModel
# from abr_speech.models import SpikingSpeechModel

n_neurons = 256
n_spiking_layers = 2

max_rate = 100
softlif_scale = 1
lif_scale = 5
base_amp = 0.01

train = False
resume = False
rate = 0.0001
n_epochs = 5

generate_nengo_params = True

tf_path = './tf_example_model'
# nengo_path = '../checkpoints/nengo_example_model'

# load the audio files collected from turkers
with open('./keyword_data.pickle', 'rb') as pfile:
    mturk = pickle.load(pfile)

n_speakers = len(mturk.speakers)
print('Speakers: %d' % n_speakers)
print('Testing Items: %d' % len(mturk.test_data))
print('Training Items: %d' % len(mturk.train_data))

# load a tensorflow model that aligns audio windows with specific chars
tf_model = TFSpeechModel(n_speakers=n_speakers, checkpoints=tf_path)
tf_model.load(tf_path)

# # define neuron models for training and for inference
# softlifs = nengo_dl.SoftLIFRate(
#     tau_rc=0.02, tau_ref=0.002, sigma=0.002, amplitude=base_amp/softlif_scale)

# lifs = nengo.LIF(
#     tau_rc=0.02, tau_ref=0.001, amplitude=base_amp/lif_scale)

# convert data to (n_items, n_steps, n_features) Nengo DL node format
nengo_train_data = tf_model.create_nengo_data(mturk.train_data, n_steps=1)
print(nengo_train_data['inp'].shape)
# nengo_test_data = tf_model.create_nengo_data(mturk.test_data, n_steps=1)

# nengo_model = SpikingSpeechModel(n_neurons=n_neurons,
#                                  n_inputs=tf_model.size_in,
#                                  n_chars=tf_model.n_chars,
#                                  n_ids=tf_model.n_speakers,
#                                  n_layers=n_spiking_layers,
#                                  checkpoints=nengo_path)

# build the network using softlifs for training
# nengo_model.build_network(softlifs, softlif_scale, max_rate)

# if train:
#   nengo_model.train(nengo_train_data, rate, n_epochs, resume=resume)
  
# if generate_nengo_params:
#   nengo_model.save_param_dict('reference_params.pickle')

# print('Train loss: %.2f' % nengo_model.compute_error_metric(nengo_train_data))
# print('Test loss: %.2f' % nengo_model.compute_error_metric(nengo_test_data))

# # rebuild the network using LIF neurons for spiking inference
# nengo_model.build_network(lifs, lif_scale, max_rate)
# nengo_model.set_probes(char_synapse=0.005, id_synapse=None, d_synapse=0.02)

# # format data as continual streams for doing evaluations
# train_stream = tf_model.create_nengo_data(
#     mturk.train_data, n_steps=10, stream=True, itemize=True)
# test_stream = tf_model.create_nengo_data(
#     mturk.test_data, n_steps=10, stream=True, itemize=True)

# # compute statistics and display example decodings
# nengo_model.compute_d_vectors(test_stream)
# nengo_model.compute_statistics(test_stream)

# print(nengo_model.stats)

# for arrays, text, speaker_id, _ in test_stream[:10]:
#     p_text = nengo_model.decode_audio(arrays)
#     print('Correct: %s' % text)
#     print('Predicted: %s' % p_text)
#     print('')
