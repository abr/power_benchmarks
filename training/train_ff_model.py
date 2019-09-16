import pickle
import string
import numpy as np

from training_models import CTCSpeechModel, FFSpeechModel
from training_utils import compute_stats, TFSample

resume = False
rate = 0.0005
n_epochs = 8

ctc_path = './checkpoints/tf_ctc_model'
ff_path = './checkpoints/tf_ff_model'

# load the audio files collected from turkers
with open('./keyword_data.pkl', 'rb') as pfile:
    dataset = pickle.load(pfile)

# load itemized train and test data for evaluation
with open('../data/test_data.pkl', 'rb') as pfile:
    test_data = pickle.load(pfile)

with open('../data/train_data.pkl', 'rb') as pfile:
    train_data = pickle.load(pfile)


n_speakers = len(dataset['speakers'])
print('Speakers: %d' % n_speakers)
print('Testing Items: %d' % len(dataset['test']))
print('Training Items: %d' % len(dataset['train']))

# load a tensorflow model that aligns audio windows with specific chars
ctc_model = CTCSpeechModel(n_speakers=n_speakers)
ctc_model.load(ctc_path)

# convert data to (n_items, n_steps, n_features) format for batch training
ff_data = ctc_model.create_nengo_data(dataset['train'], n_steps=1)

ff_model = FFSpeechModel(checkpoints=ff_path)
ff_model.train(ff_data, rate=rate, n_epochs=n_epochs, resume=resume)

# print whole-word spotting accuracy on itemized datasets
print('Training Data Statistics:')
compute_stats(ff_model, train_data)
print('')
print('Testing Data Statistics:')
compute_stats(ff_model, test_data)

ff_model.save_inference_params('../data/trained_weights.pkl')
