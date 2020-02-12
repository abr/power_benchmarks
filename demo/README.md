# Multiple Backend Keyword Spotter Demo

This is an example of a trained keyword spotter designed to recognize the word
"aloha". This demo uses a test set of samples as input instead of using al ive
microphone for ease of use.

This demo also highlights Nengo's ability to run the same frontend Python
description across multiple backends with ease!


## Installation

It is recommended to use a virtual environment such as conda, but this is not
strictly necessary. The instructions here assume conda is being used.

First create your environment under Python 3.6 and activate it.
```
conda create -n keyword_demo python=3.6
conda activate keyword_demo
```

Navigate to the ``power_benchmarks`` directory and download the data files, this
will require the ``requests`` package:
```
pip install requests
python download.py
```

Navigate to the demo directory in this repository (``power_benchmarks/demo``)
and use pip to install some requirements. (**NOTE** This will create a ``src``
directory as we are pulling some specific repository commits instead of
installing the normal distributed version from pypi.)
```
pip install -r requirements.txt
```

This will install compatible versions of ``nengo-gui``, ``nengo``,
``nengo-fpga``, and ``nengo-ocl``. It's possible the ``nengo-ocl`` install
fails, in which case please look at
[the nengo-ocl documentation](https://github.com/nengo/nengo-ocl)
for how to troubleshoot and install ``nengo-ocl``.

Now go into the nengo-fpga install and setup the config. Update
``power_benchmarks/demo/src/nengo-fpga/fpga_config`` with your FPGA details.


## Usage

In order to use ``nengo-ocl`` in the GUI, we need to set the context (device) as
an environment variable (``PYOPENCL_CTX``). Typically you will want the first
device (device 0), but if that is not the case, see below for more info.

Try launching the GUI with the first OpenCL device:
```
PYOPENCL_CTX=0 nengo keyword_demo.py
```

Once the GUI launches you can click the play button to see the model run!

**NOTE** The GUI will display ``Layer 0`` as the FPGA ensemble, This will act as
a standard ensemble unless the ``nengo-fpga`` backend is used. There will be a
warning stating this.


### Switching Backends

By default the GUI will run using the standard ``nengo`` simulator. The backend
can be easily changed in the GUI:

1. In the top left, click the _utilities_ wrench icon.
2. Click _configure Settings_.
3. Select the desired backend from the drop down menu.

**NOTE** With the ``nengo-fpga`` backend selected, there will be roughly a
10-15s pause after you click the play button. This is expected and is the
initialization and connection stage of the FPGA; unfortunately the GUI does
not display any indication of this.

**NOTE** Because of how the GUI and the ``nengo-fpga`` simulator interact, once
the ``nengo-fpga`` backend is selected, you will need to restart the GUI
before switching back to ``nengo`` or ``nengo-ocl`` backends.


### pyopencl

If using ``PYOPENCL_CTX=0`` does not work for you, it is possible to get a list
of OpenCL platforms and associated devices from the ``pyopencl`` package:

Start a Python session in your terminal (i.e. execute ``python``) and run the
following:
```
import pyopencl as cl

# To list available platforms (e.g. Nvidia, Intel, etc.)
cl.get_platforms()

# Check devices for given platform (e.g. GTX 970)
cl.get_platforms()[0].get_devices()
```
