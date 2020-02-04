import os
import pickle
import string
import numpy as np

try:
    import requests
    has_requests = True
except ImportError:
    has_requests = False
    
# permissible transcriptions for computing accuracy stats
allowed_text = ['loha', 'alha', 'aloa', 'aloh', 'aoha', 'aloha']
id_to_char = np.array([x for x in string.ascii_lowercase + '\' -'])


def create_stream(stream, dt=0.001): 

    def play_stream(t, x):
        ti = int(t / dt) - 1
        return stream[ti, :]

    return play_stream


def merge(chars):
    '''Merge repeated characters and strip blank CTC symbol'''
    acc = ["-"]
    for c in chars:
        if c != acc[-1]:
            acc.append(c)

    acc = [c for c in acc if c != "-"]
    return "".join(acc)


def weight_init(shape):
    '''Convenience function for randomly initializing weights'''
    weights = np.random.uniform(-0.05, 0.05, size=shape)
    return weights


def predict_text(sim, char_probe, n_steps, p_time):
    '''Predict a text transcription from the current simulation state'''
    n_frames = int(n_steps / p_time)
    char_data = sim.data[char_probe]
    n_chars = char_data.shape[1]

    # reshape to seperate out each window frame that was presented
    char_out = np.reshape(char_data, (n_frames, p_time, n_chars))

    # take most ofter predicted char over each frame presentation interval
    char_ids = np.argmax(char_out, axis=2)
    char_ids = [np.argmax(np.bincount(i)) for i in char_ids]

    text = merge(''.join([id_to_char[i] for i in char_ids]))
    text = merge(text)  # merge repeats to help autocorrect

    return text


def download(fname, drive_id):
    '''Download a file from Google Drive.

    Adapted from https://stackoverflow.com/a/39225039/1306923
    '''
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    url = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(url, params={'id': drive_id}, stream=True)
    token = get_confirm_token(response)
    if token is not None:
        params = {'id': drive_id, 'confirm': token}
        response = session.get(url, params=params, stream=True)
    save_response_content(response, fname)


def load(fname, drive_id):
    '''Load file either by downloading or using already downloaded version'''
    if not os.path.exists(fname):
        if has_requests:
            print("Downloading %s..." % fname)
            download(fname, drive_id)
            print("Saved %s to %s" % (fname, os.getcwd()))
        else:
            link = "https://drive.google.com/open?id=%s" % drive_id
            raise RuntimeError(
                "Cannot find '%s'. Download the file from\n  %s\n"
                "and place it in %s." % (fname, link, os.getcwd()))
    print("Loading %s" % fname)
    with open(fname, "rb") as fp:
        ret = pickle.load(fp)
    return ret


def compute_tf_stats(model, data):
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

        predicted_chars = model.merge(model.merge(''.join(chars)))

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

