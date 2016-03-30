=================
Generate Database
=================

In order to properly train the network, you need three datasets:

- A training set for building the model;
- A validation set for validating the model;
- A test set, which contains the images you want to classify.

You can build the sets yourself, or you can generate them using Google Image search.

Database format
----------------

Internally, *hacking_heat* relies on HDF5 files for storing the images and labels. The function .. py:function::pack() can create the files for you. If instead you want to build your own HDF5 files, refer to the Mocha documentation.

The easiest way to build the database is to use a file list with references to your images. The references can be local paths or URLs. The training and validation sets need furthermore the class label associated with the image.

The files, preferrably saved as *.csv*, have the following format::

<path/to/img1.jpg>,<label_of_img_1>
<path/to/img2.jpg>,<label_of_img_2>
...
<path/to/img2.jpg>,<label_of_img_n>

In the test set, omit the "," and label.

To build the database, pass this file to 

>>> pack(name, fname, img_size=(227,227), greyscale=False, flatten=False, istest=False))

Arguments::

name (str): Name of the database file to be built
fname (str): Name of the file containing the image paths
img_size (tuple of ints, optional): Size to which images will be resize. Make sure the important features will still be recognizeable in the new size, but keep in mind that larger image size will require more resources.
greyscale (bool, optional): Convert the image to greyscale. Default is False.
flaten (bool, optional): Convert the image data into a 1D array. Default is False.
istest (bool, optional): Set to True if the file is part of the test set, i.e. does not contain labels.

Example:
>>> pack('validation_db', 'validation_imgs.csv', img_size=(128, 128))

This will produce the file 'validation_db.hdf5' with all the image and label data, as well as a 'validation_db.txt' file that contains the path to the HDF5 file. The text file is a requirement for the correct working of Mocha.

Use Google
-----------

You can use Google Image search to build your training and/or validation set. This queries Google for the indicated search terms (or tags), which at the same time will serve as the labels.

>>> build_database(fname, size, tags, startIndex=0)

The function :py:func:`build_database` queries google image search for the tags and stores the urls and tags in fname.

Arguments::

	fname (str): Filename; if the file already exists, the database will be extended
	size (int): Number of images to retrieve per tag
	tags (list): List of Google search terms
	startIndex (int, optional): Index, from which image to start

Example:

>>> build_database('training_set.csv', 20, ['apples', 'oranges'], startIndex=20)

.. note:: Google API has a limitation on how many images you can retrieve, which is capped at 1000 images/tag (or something like that).

Database tools
--------------

You can store images from URL's locally using

>>> img_downloader(sourcefile, target_folder, base_folder="IMG/",
	filetype="JPEG")

Arguments::

sourcefile (str): The csv file containing the URL's
target_folder (str): The folder where to store the downloaded images. Will raise an exception if the folder does not exist
base_folder (str): If the target folder is not in the current working directory, specify its location here.
filetype (str, optional): The file format of the images, default is jpeg.

This function is handy if you built a database from google, but wish to have a local copy of your images. It automatically build a new csv with the new path information.

If you want to look at the images contained in your database, you can use

>>> show_set(fname, target)

where fname is the csv file you want to display, and target the html file it will be written to.


