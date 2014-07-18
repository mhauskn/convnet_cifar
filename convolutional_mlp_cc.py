"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
# from theano.tensor.signal import downsample
# from theano.tensor.nnet import conv
from shownet import *

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

from theano.sandbox.cuda.basic_ops import gpu_contiguous
from pylearn2.sandbox.cuda_convnet.filter_acts import FilterActs
from pylearn2.sandbox.cuda_convnet.pool import MaxPool


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, filter_pad=0, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        # conv_out = conv.conv2d(input=input, filters=self.W,
        #         filter_shape=filter_shape, image_shape=image_shape)
        input_shuffled = input.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        filters_shuffled = self.W.dimshuffle(1, 2, 3, 0) # bc01 to c01b
        conv_op = FilterActs(pad=filter_pad, stride=1, partial_sum=1)
        contiguous_input = gpu_contiguous(input_shuffled)
        contiguous_filters = gpu_contiguous(filters_shuffled)
        conv_out_shuffled = conv_op(contiguous_input, contiguous_filters)

        # downsample each feature map individually, using maxpooling
        # pooled_out = downsample.max_pool_2d(input=conv_out,
        #                                     ds=poolsize, ignore_border=True)
        pool_op = MaxPool(ds=poolsize[0], stride=poolsize[0])
        pooled_out_shuffled = pool_op(conv_out_shuffled)
        pooled_out = pooled_out_shuffled.dimshuffle(3, 0, 1, 2) # c01b to bc01

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.01, n_epochs=10000,
                    dataset='cifar-10-batches-py',
                    nkerns=[32, 64, 128], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # Example of how to reshape and display input
    # a=train_set_x[0].reshape((3,1024,1)).eval()
    # make_filter_fig(fname='results/input.png',
    #                 filters=a,
    #                 combine_chans=True)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (32, 32)  # this is the size of MNIST images
    nChannels = 3      # the number of channels

    print '... building the model'

    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    reshaped_input = x.reshape((batch_size, 3, 32, 32))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (32-5+1+4,32-5+1+4)=(32,32)
    # maxpooling reduces this further to (32/2,32/2) = (16,16)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],16,16)
    conv0 = LeNetConvPoolLayer(
        rng, input=reshaped_input,
        image_shape=(batch_size, 3, 32, 32),
        filter_shape=(nkerns[0], 3, 5, 5),
        filter_pad=2,
        poolsize=(2, 2))

    # conv0_vis = HiddenLayer(rng, input=conv0.output.flatten(2),
    #                         n_in=nkerns[0] * 16 * 16,
    #                         n_out=3 * 32 * 32, activation=T.tanh)
    # print conv0_vis.W.eval().shape # (8192, 3072)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (16-5+1+2,16-5+1+2)=(14,14)
    # maxpooling reduces this further to (14/2,14/2) = (7,7)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],7,7)
    conv1 = LeNetConvPoolLayer(
        rng, input=conv0.output,
        image_shape=(batch_size, nkerns[0], 16, 16),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        filter_pad=1,
        poolsize=(2, 2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size,128*4*4) = (batch_size,2048)
    hidden_input = conv1.output.flatten(2)

    # construct a fully-connected sigmoidal layer 
    hidden = HiddenLayer(rng, input=hidden_input, n_in=nkerns[1] * 7 * 7,
                         n_out=1024, activation=T.tanh)
    hidden_vis = HiddenLayer(rng, input=hidden.output, n_in=1024,
                             n_out=3072, activation=T.nnet.sigmoid)

    # classify the values of the fully-connected sigmoidal layer
    softmax = LogisticRegression(input=hidden.output, n_in=1024, n_out=10)
    softmax_vis = HiddenLayer(rng, input=softmax.p_y_given_x,
                              n_in=10, n_out=3072,
                              activation=T.nnet.sigmoid)

    # the cost we minimize during training is the NLL of the model
    cost = softmax.negative_log_likelihood(y)
    hidden_vis_cost = hidden_vis.reconstruction_cost(x)
    softmax_vis_cost = softmax_vis.reconstruction_cost(x)

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], softmax.errors(y),
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], softmax.errors(y),
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = softmax.params + hidden.params + conv1.params + conv0.params
    hidden_vis_params = hidden_vis.params
    softmax_vis_params = softmax_vis.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    hidden_vis_grads = T.grad(hidden_vis_cost, hidden_vis_params)
    softmax_vis_grads = T.grad(softmax_vis_cost, softmax_vis_params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    for param_i, grad_i in zip(hidden_vis_params, hidden_vis_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    for param_i, grad_i in zip(softmax_vis_params, softmax_vis_grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    print '... training'

    patience = 1000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    costs = []
    valid = []

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index

            cost_ij = train_model(minibatch_index)
            costs.append(cost_ij)

            if iter % 100 == 0:
                print('Step %d Cost %f' % (iter, cost_ij))
                make_filter_fig(fname='results/hidden.png',
                                filters=hidden_vis.W.T.eval().reshape((3,1024,1024)),
                                filter_start=0,
                                num_filters=16*16,
                                combine_chans=True)
                make_filter_fig(fname='results/softmax.png',
                                filters=softmax_vis.W.T.eval().reshape((3,1024,10)),
                                filter_start=0,
                                num_filters=10,
                                combine_chans=True)

                # rs = conv0_vis.W.reshape((3, nkerns[0] * 16 * 16, 32*32)) # (3,8192,1024)
                # rs2 = rs.dimshuffle(0,2,1)
                # make_filter_fig(fname='results/conv0.png',
                #                 filters=rs2.eval(),
                #                 filter_start=0,
                #                 num_filters=16*16,
                #                 combine_chans=True)

                # rs = conv0_vis.W.T # (3072,8192)
                # rs2 = rs.reshape((3, 1024, 8192))
                # make_filter_fig(fname='results/conv0-alt.png',
                #                 filters=rs2.eval(),
                #                 filter_start=0,
                #                 num_filters=16*16,
                #                 combine_chans=True)

            if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                valid.append(this_validation_loss * 100.)
                print('epoch %i, minibatch %i/%i, validation error %.2f%%' % \
                      (epoch, minibatch_index + 1, n_train_batches, \
                       this_validation_loss * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    best_params = params

                    # test it on the test set
                    test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)
                    print(('New Best! epoch %i, minibatch %i/%i, test error of best '
                           'model %.2f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('Optimization complete.')
    print('Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    return best_params

if __name__ == '__main__':
    params = evaluate_lenet5()
    cPickle.dump(params, open('results/params.pkl','w'))


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
