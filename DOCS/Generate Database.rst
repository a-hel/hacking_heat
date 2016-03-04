=================
Generate Database
=================

In order to properly train the network, you need three datasets:

- A training set for building the model;
- A validation set for testing the model;
- A test set, which contains the images you want to classify.

You can build the sets yourself, or you can generate them through Google like this:

Use Google
-----------

>>> import main
>>> main.build_database(fname, size, tags, startIndex=0)

The function :py:func:`build_database` queries google image search for the tags and stores the urls and tags in fname.

Arguments::
	fname (str): Filename; if the file already exists, the database will be extended
	size (int): Number of images to retrieve per tag
	tags (list): List of Google search terms
	startIndex (int, optional): Index, from which image to start

Example::
>>> build_database('training_set.csv', 20, ['apples', 'oranges'], startIndex=20)

.. note:: Google API has a daily quota of how many images you can retrieve; if you need a bigger database, run the script daily with increaseing startIndex.

Use your own files
-------------------

Create a file in the following format::

    [path to img_1],[tag_of_img_1]
    [path to img_2],[tag_of_img_2]
    ...
    [path to img_n],[tag_of_img_n]

Create the test set
-------------------

The test set, obviously, does not have any tags or labels. Just put a list of paths/urls to the file.