=============
Prerequisites
=============

Modules and packages
---------------------

*Hacking_heat* contains two modules; one to build and manipulate the image database, and one to build and train the network.

The database script is called *build_db.py* and is written in Python 2.7. If you want to use the Google image search, you need to install the Google Python API:

>>> pip install --upgrade google-api-python-client

.. Note:: The script will run without the module installed, but will raise an exeption if you try to access the image search.

.. Note:: Refer to the last section on how to set up your Google account for the Google Custom Search API.

The script to build and run the network is called *build_network.jl* and is written in Julia. You can download Julia v0.4 `Here <http://www.julialang.org>`_.

To install the `Mocha <https://devblogs.nvidia.com/parallelforall/mocha-jl-deep-learning-julia/>`_ package, use the command

>>> Pkg.add("Mocha")


Activating Google Custom Search
--------------------------------

- Go to `Google Custom Search <https://cse.google.com/cse/all>`_.
- Click on 'New search engine'
- For 'Sites to search', add something like '*.com'
- Enter a name for the search engine and click 'Create'
- Go to 'Edit search engine' and select your engine
- Activate 'Image search'
- On 'Sites to search', select 'Search the entire web but emphasize included sites'
- Get the engine's cx by clicking the 'Search engine ID'. Save the ID in a file.