*************************************
ABR Keyword Spotting Power Benchmarks
*************************************

This repository contains power benchmarking code for running a simple two-layer, 256 neuron per layer neural network keyword spotter on both neuromorphic and conventional hardware devices. On conventional devices a Tensorflow version of the keyword spotter is used, while on neuromorphic devices (Loihi), an architecturally identical Nengo version is used. This repo contains instructions for running power benchmarking trials, plotting the associated results, and reproducing our paper summarizing these results. 


**Installation**
~~~~~~~~~~~~~~~~

To start, you can make an Anaconda environment (or a virtualenv along the same lines), activate it, and install the following dependencies:

.. code-block:: batch
 
 conda create -n benchmarks python=3.6
 conda activate benchmarks
 conda install -c anaconda tensorflow-gpu
 pip install s-tui

 conda install seaborn
 conda install pandas

To install Tensorflow on Jetson: https://github.com/jetsonhacks/installTensorFlowJetsonTX

For the Movidius SDK, run the following commands, noting that the SDK doesn't always work well with either virtualenv or Anaconda, so you might need to change your path or set up a new Linux user. 

.. code-block:: batch
 
 git clone http://github.com/Movidius/ncsdk && cd ncsdk && make install

Check out the Movidius documentation to get more details. Importantly, we are using NCSDK 1.12, which has technically been superseded by NCSDK 2.0. 2.0 does introduce a queueing mechanism for passing data on and off the NCS, which could potentially yield an improvement on our reported power measurements.


**Running Experiments**
~~~~~~~~~~~~~~~~~~~~~~~

Once in this repository, the Tensorflow model code is defined in :code:`models.py`. To get network parameters and data, run :code:`python download.py` from within this directory on the command line. You may need to :code:`pip install requests` in order to perform this download. To assess the accuracy of the model on a given hardware device, run :code:`python run_accuracy_check.py --[gpu/cpu/movidius]`. If you are running  the Movidius NCS, you will have to add a path to a compiled graph with the flag :code:`--mov_graph=[graph]`. See below for instructions on compiling this graph. Running a benchmarking experiment with CPU/GPU (or compiling a Movidius graph) will write a Tensorboard summary to enable visual inspection of the network structure.   

Benchmarking and graph compilation scripts take a variety of command line options for configuring things like the batchsize, the number of neurons per layer, the number of scaling layers/branches in scaling experiments, the run time, and the log file names. If you are running new scripts, be aware of these configuration options, as they are used to organize the data during the analysis phase. 

**CPU Benchmarks**
~~~~~~~~~~~~~~~~~~

To run the CPU benchmarks, you will need to collect a log of idle baseline recordings, along with a log of runtime recordings. To collect the baseline power consumption data, run the following from a terminal:

.. code-block:: bash
 
 s-tui --csv-file [idle_dir/file_prefix.csv]

This command records a power reading at a specifiable frequency (we use 200ms; you can configure s-tui if needed) and logs them to a CSV file. To do the same thing while running the model on the CPU, setup the logger to run in the background: 

.. code-block:: bash

 s-tui --csv-file [running_dir/file_prefix.csv]

Now run the model from a separate terminal: 

.. code-block:: bash

 python run_benchmark.py --cpu --bsize=1 --time=900 --log=[running_dir/file_prefix.json]

Once this script terminates, close the logger in the other terminal. Note that a separate log is created by the benchmark script - this specifies metadata such as the hardware being used, the duration of the run, the total number of inferences, and timestamps for entering and exiting the inference loop. Be careful to ensure the file prefix is the same for both the power log and the metadata log, as this prefix will be used to collate the metadata with each power reading to create a collection of data samples during the analysis phase. Also be sure to include the name of the hardware device in lowercase in the file prefix, since filenames will be used to match runtime logs to the correct baseline log to compute idle power consumption.

Note that command line arguments for the benchmarking script can specify the following data attributes: hardware type, runtime, batchsize, number of model copies, multiplier on hidden layer neuron count, and the log file. During analysis each recorded power measurement is collated with this data in a Pandas dataframe to support arbitrary queries and visualizations. Movidius and Loihi only support a batchsize of 1. By default, batchsize, number of model copies, and the neuron count multiple are all set to 1. To reproduce the scaling experiments in the paper, add :code:`--n_layers=10` and :code:`--n_copies=[N]` before executing `run_benchmark.py`. The same configurations apply for subsequent hardware devices.

**GPU Benchmarks**
~~~~~~~~~~~~~~~~~~

To run the GPU benchmarks, the same process as above is used, just with a different logging command: 

.. code-block:: bash

 nvidia-smi -i 0 -f [idle_dir/filename] --loop-ms=200 --format=csv --query-gpu=timestamp,power.draw

This command records a power reading at a specifiable frequency and logs them to a CSV file. To do the same thing while running the model on the GPU, setup the logger to run in the background as before: 

.. code-block:: bash

 nvidia-smi -i 0 -f [running_dir/file_prefix.csv] --loop-ms=200 --format=csv --query-gpu=timestamp,power.draw
 
Now run the model from a separate terminal: 

.. code-block:: bash

 python run_benchmark.py --gpu --bsize=1 --time=900 --log=[running_dir/file_prefix.json]

Once this script terminates, close the logger in the other terminal. Again, a separate log is created by the benchmark script so be careful as before to ensure the file prefix is the same for both the power log and this metadata log.

One important point regarding memory allocation on the GPU: Tensorflow by default allocates 100% of GPU memory regardless of the nature of the underlying computational graph. To safeguard against this distorting power measurements, the default per process gpu memory fraction in tensorflow is set to 0.1.


**Jetson Benchmarks**
~~~~~~~~~~~~~~~~~~~~~

Jetson TX1 benchmarks need to be run directly on the Jetson board via SSH. The methodology for computing the power load is the same as before, but depending hardware version, it may not possible to record power consumption using software (see `here <https://goo.gl/bPzwYX>`_ for details). We use an Intertek P4455 power monitoring device to observe the power drawn by the board from the wall socket under idling and runtime conditions. Because no automatic logging is available in this context, the estimated consumption is less precise here. 

For consistency with previous experiments and for later analysis, log the idle power consumption level in :code:`[idle_dir/jetson.csv]`. Next, run the benchmark script as follows while carefully observing the power monitor:

.. code-block:: bash
 
 python run_benchmark.py --gpu --bsize=1 --time=900 --log=[running_dir/file_prefix.json]
 
Note that this is the same command used to run on a regular GPU device. While the inference loop is running, record the power consumption levels in `[running_dir/jetson.csv]`. 

**Movidius Benchmarks**
~~~~~~~~~~~~~~~~~~~~~~~

The same technique is again used to estimate power consumption, but since the NCS plugs into a USB port, an inline voltage and current meter (https://www.adafruit.com/product/1852) is used to measure idle and runtime power consumption for the NCS device. As before, no automatic logging is available in this context, so the estimated consumption is less precise than in the first two experiments (though the displayed values on the USB meter are quite stable over each of the idling and runtime periods). Note that the Movidius NCS does not currently support batched inference, so it is only possible to perform experiments using a batchsize of one. See https://ncsforum.movidius.com/discussion/881/queueing-multiple-input-tensors for details.

To run the benchmark, the Tensorflow model used in previous experiments will first need to be compiled into a graph compatible with the NCS:

.. code-block:: bash

   python make_movidius_graph.py --ckpt=[ckpt_prefix] --nx_neurons=1 [--scaled --n_layers=[n] --n_copies=[c]]

This script uses a specified model checkpoint name to create a compiled graph to run on the NCS, containing the specified multiple of hidden layer neurons (if this is 1, the keyword spotting weights are loaded). If the `--scaled` flag is set, the number of copies and layers should be set to get the desired scaling configuration. A maximum of 12 SHAVEs are used in the compilation as per the documentation examples at https://movidius.github.io/ncsdk/tf_compile_guidance.html. 

Once a compiled graph is available, run the following command while monitoring the USB meter:
 
.. code-block:: bash

 python run_benchmark.py --movidius --mov_graph=[graph] --time=900 --log=[running_dir/file_prefix.json]

Set the flags for the benchmarking script to be equal to those of in the graph creation script to ensure that the resulting data is tagged with the right configuration labels (n_copies, n_layers, b_size, etc.). Finally, it is important to note that the USB meter measures voltage and current, so these values need to be multiplied to compute the power load. This is straightforward to accomplish if observed recordings are logged into spreadsheet software. Note that because the analysis scripts are expecting s-tui formatted logs, the power recordings should be enter into the 8th column of the resulting csv file. 


**Loihi Benchmarks**
~~~~~~~~~~~~~~~~~~~~

To run Loihi scriptis you will need to install `nengo-loihi` by following the instructions in the documentation `here <https://www.nengo.ai/nengo-loihi/installation.html>`_. The script we will use adapts this `example <https://www.nengo.ai/nengo-loihi/examples/keyword_spotting.html>`_ to use a longer series of inputs to enable sustained measurements of power consumption. We use the LT Powerplay app and measurement device to log values during runtime. 

Our benchmarking scripts contain unreleased, proprietary code and are run on a research chip that is not publically available. As such, please contact us directly if you have access to the code and a Wolf Mountain board and would like to replicate our reported results.

**Results**
~~~~~~~~~~~

To plot the results of the experiments and replicate the graphs in the paper, run the following command form the root of this benchmarking directory:

.. code-block:: bash
   
  python analysis/summary.py --idle_dir=./logs/idle --running_dir=./logs/running

This will display a series of plots and write them to file for inclusion the paper when it compiled to a PDF. To compile the paper, do the following:
  
.. code-block:: bash

    cd paper/ 
    pdflatex power_summary.tex
    bibtex power_summary
    pdflatex power_summary.tex
    pdflatex power_summary.tex

This should create a pdf in the `papers` directory for your perusal.

