=============
Build Network
=============

Build and train the network as follows:

>>> import main
>>> main.build_network(f_train, f_val, f_test, img_size, greyscale=False, flatten=False, num_epochs=500)

Arguments::

	f_train (str): path to the training set file
	f_val (str): path to the validation set file
	f_test (str): path to the test set file
	img_size (tuple): size to which images will be resized
	greyscale (bool, optional): if True, images will be converted to greyscale
	flatten (bool, optional): if True, images will be flattened
	architecture (str, optional): Either 'mlp' (multi-layer perceptron)
		or 'cnn' (Convoluted neural network)
	num_epochs (int, optional): Number of training cycles

Example:

>>> build_network('training.csv', 'validation.csv', 'test.csv', (128,128), num_epochs=100)