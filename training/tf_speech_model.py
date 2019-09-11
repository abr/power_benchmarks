import pickle
from rate_model import TFSpeechModel
from utils import TurkHandler

train = False
resume = False
rate = 0.0005
n_epochs = 3
load_data = False
checkpoints = './tf_example_model'

allowed_text = ['loha', 'alha', 'aloa', 'aloh', 'aoha', 'aloha']

if load_data:
    # load the audio files collected from turkers
    # set these to set up data saving and loading
    turkpath = './all_data/'  # point to wherever data is
    id_file = './96_speaker_id_list.pickle'

    mturk = TurkHandler(turkpath, id_file=id_file)
    mturk.load_data(n_clean=1, pos_neg_ratio=4)

    n_speakers = len(mturk.speakers)
    
    with open('./keyword_data.pickle', 'wb') as pfile:
        pickle.dump(mturk, pfile)
else:
    with open('./keyword_data.pickle', 'rb') as pfile:
        mturk = pickle.load(pfile)


n_speakers = len(mturk.speakers)
print('Speakers: %d' % n_speakers)
print('Testing Items: %d' % len(mturk.test_data))
print('Training Items: %d' % len(mturk.train_data))


# train the model
tf_model = TFSpeechModel(n_speakers=n_speakers, checkpoints=checkpoints)

if train:
    tf_model.train(mturk.train_data, rate=rate, n_epochs=n_epochs, resume=resume)
else:
    tf_model.load(checkpoints)

# compute some transcription stats after training
print('Pos char LER on train data:')
pos_data = [x for x in mturk.train_data if x.text == 'aloha']
print('LER: %4f' % tf_model.label_error_rate(pos_data))
print('')

print('Neg char LER on train data:')
neg_data = [x for x in mturk.train_data if x.text != 'aloha']
print('LER: %4f' % tf_model.label_error_rate(neg_data))
print('')

print('Pos char LER on test data:')
pos_data = [x for x in mturk.test_data if x.text == 'aloha']
print('LER: %4f' % tf_model.label_error_rate(pos_data))
print('')

print('Neg char LER on test data:')
neg_data = [x for x in mturk.test_data if x.text != 'aloha']
label_error = tf_model.label_error_rate(neg_data)
print('LER: %4f' % label_error)
print('')

# compute some speaker identification stats after training
train_id_error = tf_model.id_error_rate(mturk.train_data)
print('ID error on train data: %4f' % train_id_error)

test_id_error = tf_model.id_error_rate(mturk.test_data)
print('ID error on test data: %4f' % test_id_error)

# print some examples transcriptions
for sample in mturk.test_data[:5]:
    predicted_chars = tf_model.predict_chars(sample.audio)
    print('Correct: %s' % sample.text)
    print('Predicted: %s' % predicted_chars)
    print('')


# print whole-word spotting accuracy
correct = 0
count = 0
for sample in mturk.test_data:
    predicted_chars = tf_model.predict_chars(sample.audio)
    if predicted_chars in allowed_text and sample.text == 'aloha':
        correct += 1
    if sample.text == 'aloha':
        print(predicted_chars)
        count += 1

print('Accuracy: %4f' % (correct / count))

tf_model.save_inference_params('./test_weights.pkl')
