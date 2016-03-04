Prerequisites
=============

Hacking_heat runs with Python 2.7. It uses Theano to build and train networks and lasagna to communicate with Theano.

Install Theano:

>>> pip install theano

Install lasagna:

>>> pip install lasagna

In order to use Google image search, you need to install the Google Python API:

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