import pickle
from training_models import CTCSpeechModel
from training_utils import TFSample, allowed_text

train = False
resume = False
rate = 0.0005
n_epochs = 2
load_data = False
checkpoints = "./checkpoints/tf_ctc_model"


with open("keyword_data.pkl", "rb") as pfile:
    dataset = pickle.load(pfile)

n_speakers = len(dataset["speakers"])
print("Speakers: %d" % n_speakers)
print("Testing Items: %d" % len(dataset["test"]))
print("Training Items: %d" % len(dataset["train"]))


# train the model
tf_model = CTCSpeechModel(n_speakers=n_speakers, checkpoints=checkpoints)

if train:
    tf_model.train(dataset["train"], rate=rate, n_epochs=n_epochs, resume=resume)
else:
    tf_model.load(checkpoints)

# compute some transcription stats after training
print("Pos char LER on train data:")
pos_data = [x for x in dataset["train"] if x.text == "aloha"]
print("LER: %4f" % tf_model.label_error_rate(pos_data))
print("")

print("Neg char LER on train data:")
neg_data = [x for x in dataset["train"] if x.text != "aloha"]
print("LER: %4f" % tf_model.label_error_rate(neg_data))
print("")

print("Pos char LER on test data:")
pos_data = [x for x in dataset["test"] if x.text == "aloha"]
print("LER: %4f" % tf_model.label_error_rate(pos_data))
print("")

print("Neg char LER on test data:")
neg_data = [x for x in dataset["test"] if x.text != "aloha"]
label_error = tf_model.label_error_rate(neg_data)
print("LER: %4f" % label_error)
print("")

# compute some speaker identification stats after training
train_id_error = tf_model.id_error_rate(dataset["train"])
print("ID error on train data: %4f" % train_id_error)

test_id_error = tf_model.id_error_rate(dataset["test"])
print("ID error on test data: %4f" % test_id_error)

# print some examples transcriptions
for sample in dataset["test"][:5]:
    predicted_chars = tf_model.predict_chars(sample.audio)
    print("Correct: %s" % sample.text)
    print("Predicted: %s" % predicted_chars)
    print("")


# print true positive whole-word spotting accuracy
correct = 0
count = 0
for sample in dataset["test"]:
    predicted_chars = tf_model.predict_chars(sample.audio)
    if predicted_chars in allowed_text and sample.text == "aloha":
        correct += 1
    if sample.text == "aloha":
        print(predicted_chars)
        count += 1

print("True positive accuracy: %4f" % (100 * correct / count))
