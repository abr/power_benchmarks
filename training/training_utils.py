import os
import uuid
import pickle
import random
# import librosa
import numpy as np

from collections import namedtuple


TFSample = namedtuple(
    'TFSample', ['audio', 'text', 'speaker_id', 'sample_id', 'wav_path'])

NengoSample = namedtuple(
   'NengoSample', ['arrays', 'text', 'speaker_id', 'sample_id'])

# permissible transcriptions for computing accuracy stats
allowed_text = ['loha', 'alha', 'aloa', 'aloh', 'aoha', 'aloha']


class TurkHandler(object):
    """Groups data collected from MTurk into train and test sets. Also performs
    simple splitting and trimming to extract target phrases from audio files.

    Parameters:
    -----------
    path: str
        Path to a directory containing speaker-sorted wavfiles.
    id_file: str (optional)
        Path for pickle file containing speaker ids to use.
    noise_file: str (optional)
        Path for wavfile containing background noise for data augmentation.
    """
    def __init__(self, path, id_file=None, noise_file=None):
        self.path = path
        self.id_filter = None
        self.noise = None

        if noise_file is not None:
            self.noise, _ = librosa.load(noise_file, sr=16000)

        # use list of approved IDs to filter unwanted data from training set
        if id_file is not None:
            with open(id_file, 'rb') as pfile:
                self.id_filter = pickle.load(pfile)

    def file_stream(self, normalized=False):
        '''Stream wav paths from directory containing a folder per speaker'''
        speakers = [s for s in os.listdir(self.path) if not s.startswith('.')]
        speakers = [s for s in self.id_filter] if self.id_filter else speakers
        speakers.sort()
        self.speaker_list = speakers

        for speaker in speakers:
            print('Collecting audio for speaker %s' % speakers.index(speaker))
            print('Tag: %s' % speaker)

            if normalized:
                s_path = os.path.join(self.path, speaker, 'normalized')
            else:
                s_path = os.path.join(self.path, speaker)

            wav_paths = [os.path.join(s_path, f) for f in os.listdir(s_path)]
            wav_paths = [f for f in wav_paths if f.endswith('.wav')]

            for wav_path in wav_paths:
                yield wav_path, speaker

    def normalize_audio(self):
        '''Create normalized audio files in subdir of each speaker directory'''
        for wav_path, _ in self.file_stream():
            wav_dir = '/'.join(wav_path.split('/')[:-1])
            wav_name = wav_path.split('/')[-1]
            new_path = os.path.join(wav_dir, 'normalized', wav_name)

            cmd = 'ffmpeg-normalize ' + str(wav_path) + ' -o ' + str(new_path)
            os.system(cmd)

    def load_data(self, n_clean=1, n_noisy=0, pos_neg_ratio=1, normalized=True):
        '''Collect audio data with option of augmented noisy copies'''
        if n_noisy != 0 and self.noise is None:
            raise Exception('Cannot create noisy data without noise wav file!')

        if pos_neg_ratio < 1:
            raise Exception('Specified ratio is not possible given the data.')

        self.speakers = set()
        data = list()

        for wav_path, speaker in self.file_stream(normalized=normalized):
            if speaker not in self.speakers:
                self.speakers.add(speaker)

            speaker_id = len(self.speakers) - 1  # - 1 to use 0 indexing

            filename = wav_path.split('/')[-1]
            filename_segments = filename.split('-')

            # slice from naming scheme, special case if 'test' in file name
            text = ' '.join(filename_segments[1:-6])
            phrase = ' '.join(text.split()[1:]) if 'test' in text else text

            raw_audio, _ = librosa.load(wav_path, sr=16000)

            # skip clips that are too short for typical feature window size
            if len(raw_audio) < 2500:
                continue

            # create n clean copies of each audio sample
            for idx in range(n_clean):
                audio = self.filter_phrase(raw_audio, noise=False)
                sample_id = uuid.uuid4()
                sample = TFSample(
                    audio, phrase, speaker_id, sample_id, wav_path)

                data.append(sample)

            # create n noisy versions of each audio sample
            for idx in range(n_noisy):
                audio = self.filter_phrase(raw_audio, noise=True)
                sample_id = uuid.uuid4()
                sample = TFSample(
                    audio, phrase, speaker_id, sample_id, wav_path)

                data.append(sample)

        # seperate out positive and negative examples of the target phrase
        pos_data = [d for d in data if d.text == 'aloha']
        neg_data = [d for d in data if d.text != 'aloha']

        assert len(pos_data) + len(neg_data) == len(data)

        # calculate how many negative examples needed to achieve desired ratio
        n_negs_total = round(len(pos_data) / pos_neg_ratio)
        n_negs_per_speaker = round(n_negs_total / len(self.speakers))

        self.train_data = []
        self.test_data = []

        for idx, speaker in enumerate(self.speakers):
            speaker_samples = [x for x in data if x.speaker_id == idx]
            pos_samples = [x for x in speaker_samples if x.text == 'aloha']
            neg_samples = [x for x in speaker_samples if x.text != 'aloha']

            # shuffle to get random choices for test items
            random.shuffle(pos_samples)
            random.shuffle(neg_samples)

            # pop off test items so they don't get inlcuded in training data
            self.test_data += [pos_samples.pop(), neg_samples.pop()]
            self.train_data += neg_samples[:n_negs_per_speaker]
            self.train_data += pos_samples

        # check that there are exactly two test items per speaker
        assert len(self.test_data) == 2 * len(self.speakers)

    def filter_phrase(self, raw_audio, noise=False):
        '''Break audio into non-silent chunks and extract chuck with phrase'''
        chunks = librosa.effects.split(raw_audio, top_db=35)

        # case where there is e.g. a click at the beginning of the audio
        if len(chunks) > 1:
            spans = [chunks[i][1]-chunks[i][0] for i in range(len(chunks))]
            idx = spans.index(max(spans))  # assume longest span is the phrase
        else:
            idx = 0

        audio = raw_audio[chunks[idx][0]:chunks[idx][1]]
        audio, _ = librosa.effects.trim(audio, top_db=35)

        # add noise from noise wav file
        if noise:
            noise_idx = random.randint(0, len(self.noise) - len(audio))
            noise_slice = self.noise[noise_idx:noise_idx + len(audio)]
            audio = 0.85 * audio + 0.15 * noise_slice  # sets vol of noise

        return audio

    def save_audio(self, path):
        '''Save collected audio samples to file for manual inspection'''
        for item in self.train_data + self.test_data:
            write_path = os.path.join(path, str(item.sample_id) + '.wav')
            librosa.output.write_wav(write_path, item.audio, sr=16000)


def build_arrays(ce_data, n_steps=1, stream=False):
    '''Convert CE formatted data to arrays for use with Nengo DL nodes'''
    arrays = {}
    inp = np.vstack([item['inp'] for item in ce_data])[:, None, :]
    out = np.vstack([item['out'] for item in ce_data])[:, None, :]
    ids = np.vstack([item['ids'] for item in ce_data])[:, None, :]

    if stream:
        # stack tiled arrays horizontally
        arrays['inp'] = np.hstack(np.tile(d, (1, n_steps, 1)) for d in inp)
        arrays['out'] = np.hstack(np.tile(d, (1, n_steps, 1)) for d in out)
        arrays['ids'] = np.hstack(np.tile(d, (1, n_steps, 1)) for d in ids)
    else:
        # stack tiled arrays vertically
        arrays['inp'] = np.tile(inp, (1, n_steps, 1))
        arrays['out'] = np.tile(out, (1, n_steps, 1))
        arrays['ids'] = np.tile(ids, (1, n_steps, 1))

    return arrays


def convert_to_onehot(array):
    '''Convert rows in 2D array to onehot vecs corresponding to max values'''
    argmaxs = np.argmax(array, axis=1)
    onehots = np.zeros(array.shape)
    onehots[range(len(argmaxs)), argmaxs] = 1

    return onehots


def merge(chars):
    '''Merge repeated characters and strip blank CTC symbol'''
    acc = ['-']
    for c in chars:
        if c != acc[-1]:
            acc.append(c)

    acc = [c for c in acc if c != '-']
    return ''.join(acc)


def normalize(v):
    """Normalize a vector to unit length"""
    if np.linalg.norm(v) > 0:
        return v / np.linalg.norm(v)
    else:
        return v


def with_graph(method):
    """Decorator for easily adding tensorflow ops to an object's TF graph"""
    def add_to_graph(self, *args, **kwargs):
        with self.graph.as_default():
            return method(self, *args, **kwargs)

    return add_to_graph


def compute_stats(model, data):
    '''Compute True/False Pos/Neg stats for Tensorflow keyword model'''
    stats = {
        "fp":0,
        "tp":0,
        "fn":0,
        "tn":0,
        "aloha": 0,
        "not-aloha": 0
    }

    for features, text in data:
        inputs = np.squeeze(features)

        chars = []
        for window in inputs:
            char = model.predict_text(np.expand_dims(window, axis=0))
            chars.append(char)

        predicted_chars = merge(merge(''.join(chars)))

        if text == 'aloha':
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
