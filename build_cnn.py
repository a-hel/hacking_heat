# http://ip.cadence.com/uploads/901/cnn_wp-pdf
# http://cs231n.stanford.edu/
# http://web.engr.illinois.edu/~slazebni/spring14/lec24_cnn.pdf

from __future__ import print_function

import sys
import os
import time
import numpy as np
import theano
import theano.tensor as T

import lasagne

def build_cnn(input_var=None, img_z=1, img_x=28, img_y=28,
	n_classes=2):
    """ Build a convoluted neuronal network using lasagne.
    Arguments:

    input_var: Must be 'None'
    img_z, img_x, img_y (int): Shape of single image
    n_classes (int, default=2): Number of classes to distinguish

    Returns: The network
    """

    # Input layer; shape corresponds image size:
    l_in = lasagne.layers.InputLayer(shape=(None, img_z, img_x, img_y),
                                        input_var=input_var)
    # First 2D-convolutional layer, coarse selection
    l_conv1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=4, filter_size=(12, 12),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Pooling layer 1
    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(4, 4))
    # Second conv layer, medium selection
    l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1, num_filters=16, filter_size=(8, 8),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Pooling layer 2
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(2, 2))
    # Third conv layer, fine selection
    l_conv3 = lasagne.layers.Conv2DLayer(
            l_pool2, num_filters=72, filter_size=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Pooling layer 3
    l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=(2, 2))
    # Dropout layer, followed by a dense layer
    l_dense1 = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_pool3, p=.5),
            num_units=3308,
            nonlinearity=lasagne.nonlinearities.rectify)
    # Fully connected dense layer
    l_dense2 = lasagne.layers.DenseLayer(
            l_dense1,
            num_units=1654,
            nonlinearity=lasagne.nonlinearities.rectify)
    # Fully connected dense layer
    l_dense3 = lasagne.layers.DenseLayer(
            l_dense2,
            num_units=826,
            nonlinearity=lasagne.nonlinearities.rectify)
    # Output layer, num_units=n_classes
    l_out = lasagne.layers.DenseLayer(
            l_dense2,
            num_units=n_classes,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out


def main(X_train, y_train, X_val, y_val, X_test, 
    channels, size, num_epochs=2):
    """ Build, train and apply model.

    Arguments:
    X_train (np.array): Training set images in a n*z*x*y matrix
    y_train (np.array): Classes of training set in an n array
    X_val (np.array): Validation set images in a n*z*x*y matrix
    y_val (np.array): Classes of validation set in an n array
    X_test (np.array): Test set images in a n*z*x*y matrix
    channels (int): Color channels, 3 for RGB, 1 for b/w
    size (tuple): Image size (x*y)
    num_epochs (int): Number of training cycles 

    Returns: The predictions for the test set
    """
    
    img_x, img_y, img_z = X_train.shape
    n_classes = np.unique(y_train).shape[0]

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')


    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var, img_z, img_x, img_y, n_classes)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.001, momentum=0.8)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        start_time = time.time()
        train_err += train_fn(X_train, y_train)


        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        err, acc = val_fn(X_val, y_val)
        val_err += err
        val_acc += acc

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err))
        print("  validation loss:\t\t{:.6f}".format(val_err))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0


    ######### Making predictions

    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    predict_fn = theano.function([input_var], T.argmax(test_prediction, axis=1))
    return predict_fn(X_test)
    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)