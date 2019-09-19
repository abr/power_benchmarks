import pickle
import os

import numpy as np
import tensorflow as tf

from models import TensorflowModel


frozen_graph_name = './checkpoints/benchmark_model.pb'
tflite_model_name = './checkpoints/benchmark_model.tflite'

out_nodes = ['copy_0/char_output/outputs']    # Output nodes
inp_nodes = ['inputs']

with open('./data/train_data.pkl', 'rb') as pfile:
    train_data = pickle.load(pfile)

with open('./data/inference_weights.pkl', 'rb') as pfile:
    weights = pickle.load(pfile)

# build the model using weights from previously trained model
model = TensorflowModel(n_inputs=390, n_layers=2)
model.build(n_copies=1)
model.start_session()
model.set_weights(weights)

# Freeze the graph
frozen_graph_def = tf.graph_util.convert_variables_to_constants(
    model.sess,
    model.sess.graph_def,
    out_nodes)

# Save the frozen graph as a pb file
with open(frozen_graph_name, 'wb') as f:
  f.write(frozen_graph_def.SerializeToString())


# define a generator to yield training items for quantization calibration
def representative_dataset_gen():
    for features, text in train_data:
        inputs = np.squeeze(features)
        for window in inputs:
            yield [np.expand_dims(window, axis=0)]


# use TFLiteConverter to create TFLite version of the model
converter = tf.lite.TFLiteConverter.from_frozen_graph(
   frozen_graph_name, inp_nodes, out_nodes)

converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
# converter.inference_type = tf.uint8
# converter.default_ranges_stats = (-70, 70)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
input_arrays = converter.get_input_arrays()
# use mean and as per TFLite quantization spec
converter.quantized_input_stats = {input_arrays[0] : (0, 1.452)} 

# convert the model and save to file
tflite_quant_model = converter.convert()
open(tflite_model_name, "wb").write(tflite_quant_model)

# compile the model to the edgetpu 
os.system('edgetpu_compiler ' + tflite_model_name)
