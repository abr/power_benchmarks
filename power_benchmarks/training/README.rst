*************************************
ABR Keyword Spotting Training Scripts
*************************************

This directory contains scripts for training the keyword spotter from scratch. This is a two-step process given the end-goal of running a version of the trained model on a neuromorphic hardware device. In the first step, a standard tensorflow model is trained from audio/text pairs using the CTC objective function to learning an alignment between windows of the audio signal and particular alphabetical characters. In the second step, this learned alignment is used to create training data that matches input feature windows to alphabetical characters directly. With directly aligned data of this sort, a simple feedforward model of the sort used in the benchmarking experiments described elsewhere in this repository can be trained. Note that for benchmarking purposes, the only model of direct interest is the feedforward inference model.

**Installation**
~~~~~~~~~~~~~~~~

No additional installation dependencies are required to perform model training.

**Train the CTC Alignment Model**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train this model, run the following command from terminal:

.. code-block:: bash
 
 python train_ctc_model.py

If you open the script, you can change various parameters such as the learning rate, number of epochs, and checkpoint file. Currently, a checkpoint file corresponding to the trained model used in the benchmarking experiments is included in this directory.

**Train the feedforward inference model**
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train this model, run the following command from terminal:

.. code-block:: bash
 
 python train_ff_model.py

Again, if you open the script, you can change various parameters such as the learning rate, number of epochs, and checkpoint file. By default, the trained model weights, will be saved in the `data`  directory at the root of this repo. Changing the name of weights file used in the benchmarking scripts will thus run the benchmarks with these newly trained model weights. 

Finally, because the data used in the benchmarking experiments has been conveniently formatted ahead of time so it doesn't need to be regenerated from the raw audio via feature transformations, a script is included here to validate that the datasets in fact include identical items. 
