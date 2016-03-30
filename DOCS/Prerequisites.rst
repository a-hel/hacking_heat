=============
Prerequisites
=============

*Hacking_heat* contains two modules, one to build and train the network (build_network.jl) and one to prepare the data (build_db.py).
 
If you want to prepare your data manually, you do not necessarily need the build_db.py script. However, it is recommended as it assures the correct format of the data.

To build and train the network, *hacking_heat* uses the juila package 'Mocha'. If you do not have julia installed, you can download it from the project homepage `www.julialang.org <http://www.julialang.org>`_.

From the julia command line, install Mocha:

>>> Pkg.add("Mocha")

This conveniently installs all dependencies, most importantly HDF5 support.



If you want to use Google image search to build your image database, you need to install the Google Python API:

>>> pip install --upgrade google-api-python-client

You furthermore need a Google account, where you have to activate the CustomSearch API:

- Go to `Google Custom Search <https://cse.google.com/cse/all>`_.
- Click on 'New search engine'
- For 'Sites to search', add something like '*.com'
- Enter a name for the search engine and click 'Create'
- Go to 'Edit search engine' and select your engine
- Activate 'Image search'
- On 'Sites to search', select 'Search the entire web but emphasize included sites'
- Get the engine's cx by clicking the 'Search engine ID'. Save the ID in a file.

Cuda support
-------------

Mocha.jl supports the use of Cuda, which uses your machines GPU to accelerate the training of your network. In order to install and use it, please refer to the CUDA documentation: `https://developer.nvidia.com/cuda-toolkit <https://developer.nvidia.com/cuda-toolkit>`_.
