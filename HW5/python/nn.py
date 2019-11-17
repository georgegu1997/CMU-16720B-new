import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

class TwoLayerNet():
    def __init__(self, input_size, hidden_size, output_size, params={}):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        # initialize with the give weights
        if len(params) > 0:
            self.params = params
        # random initialize the weights
        else:
            initialize_weights(input_size, hidden_size, params, "layer1")
            initialize_weights(hidden_size, output_size, params, "output")

    def predict(self, x):
        h1 = forward(x, self.params, 'layer1') # First layer
        probs = forward(h1, self.params, 'output', softmax) # Second layer
        return probs

    def loss(self, x, y):
        probs = self.predict(x)
        loss, acc = compute_loss_and_acc(y, probs)
        return loss, acc

    def backward(self, probs):
        delta1 = probs.copy()
        delta1[np.arange(probs.shape[0]), yb.argmax(axis=1)] -= 1
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b.
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    W = np.random.uniform(-np.sqrt(6)/np.sqrt(in_size+out_size), np.sqrt(6)/np.sqrt(in_size+out_size), size=(in_size, out_size))
    b = np.zeros(out_size)
    ##########################

    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None

    ##########################
    ##### your code here #####
    res = 1 / (1+np.exp(-x))
    ##########################

    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    ##########################
    ##### your code here #####
    XW = X.dot(W)
    pre_act = XW + b.reshape((1,-1))
    post_act = activation(pre_act)
    ##########################

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    res = None

    ##########################
    ##### your code here #####
    c = - x.max(axis=1, keepdims=True)
    xc = x + c
    num = np.exp(xc)
    den = num.sum(axis=1, keepdims=True)
    res = num / den
    ##########################

    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    # calculate the accuracy
    pred_label = probs.argmax(axis=1)
    label = y.argmax(axis=1)
    acc = (pred_label == label).sum() / label.shape[0]

    # calculate the cross-entropy loss
    logf = np.log(probs)
    ylogf = y * logf
    sumylogf = ylogf.sum()
    loss = - sumylogf
    ##########################

    return loss, acc

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name] # (D, C)
    b = params['b' + name] # (C, )
                           # X (N, D)
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    '''suppose the delta is dloss/dpost_act'''
    # post_act = sigmoid(pre_act)
    dpre_act = delta * activation_deriv(post_act) # (N, C)
    # pre_act = XW + b.reshape((1,-1))
    db = dpre_act.sum(axis=0) # (C,)
    dXW = dpre_act # (N, C)
    # XW = X.dot(W)
    dX = dXW.dot(W.T) # (N, D)
    dW = X.T.dot(dXW) # (D, C)

    grad_W = dW
    grad_b = db
    grad_X = dX
    ##########################

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    N = x.shape[0]
    permuted_idx = np.random.permutation(np.arange(N))

    batches = []
    for i in range(int(np.ceil(N/batch_size))):
        batch_idx = permuted_idx[i*batch_size:min((i+1)*batch_size, N)]
        batch = (x[batch_idx], y[batch_idx])
        batches.append(batch)

    ##########################
    return batches
