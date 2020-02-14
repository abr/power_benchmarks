# Multiple Backend Keyword Spotter Demo

This is an example of a trained keyword spotter designed to recognize the word
"aloha". This demo uses a test set of samples as input instead of using al ive
microphone for ease of use.

This demo also highlights Nengo's ability to run the same frontend Python
description across multiple backends with ease!


## Installation

First download the source from the ``power_benchmarks`` repository, we are using
the ``fpga`` branch:
```
git clone https://github.com/abr/power_benchmarks.git --branch fpga
```

It is recommended to use a virtual environment such as conda, but this is not
strictly necessary. The instructions here assume conda is being used.

Create your environment under Python 3.6 and activate it.
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
``nengo-fpga``, ``nengo-dl``, and ``nengo-ocl``.

Now go into the nengo-fpga install and setup the config. Update
``power_benchmarks/demo/src/nengo-fpga/fpga_config`` with your FPGA details.


### Likely Complications

#### Backends

It's possible the ``nengo-ocl`` install fails, in which case please look at [the
nengo-ocl documentation](https://github.com/nengo/nengo-ocl) for how to
troubleshoot and install ``nengo-ocl``.

Similarly, there may be modifications and torubleshooting required for
``nengo-dl``. This demo uses Tensorflow 1.12 (for CUDA 9.0). You may need to
modify this to match the CUDA drivers on your machine. See
[the nengo-dl documentation](https://www.nengo.ai/nengo-dl/)
for more information.

You can still use the other backends (e.g. ``nengo``, ``nengo-fpga``) even if
some backends fail to correctly install.


#### Audio playback

It's possible ``simpleaudio`` fails to install due to dependencies. For Linux,
try installing the following with your package manager then try installing
``simpleaudio`` again:

```
sudo apt install -y python3-dev libasound2-dev
```

You can also try installing from conda instead:
```
conda install -c skmad simpleaudio
```

If this is giving you too much trouble, you can also comment out the audio. In
the ``keyword_demo.py`` file, comment out the following two lines:
```
7   import simpleaudio as sa

...

182             _ = sa.WaveObject.from_wave_file(audio_file).play()
```


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
