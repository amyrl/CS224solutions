import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad, f_sigmoid
from q2_gradcheck import gradcheck_naive



def forward_backward_prop(data, labels, params, dimensions):
    """ 
    Forward and backward propagation for a two-layer sigmoidal network 
    
    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    ### YOUR CODE HERE: forward propagation
    z1 = data.dot(W1)+b1
    hidden = sigmoid(z1) 
    z2 = hidden.dot(W2)+b2
    y_hat = softmax(z2)
    #sum of diagonal of dot product
    cost = - np.sum(np.diagonal(labels.dot(np.log(y_hat).transpose())))
    ### END YOUR CODE
    
    ### YOUR CODE HERE: backward propagation
    delta1 = y_hat - labels #grad CE w.r.t. Z2
    gradW2 = hidden.transpose().dot(delta1) #since grad Z2 w.r.t. W2 is hidden'
    assert  W2.shape == gradW2.shape, "W2 grad mismatch"
    gradb2 = np.ones((z2.shape[0],1)).transpose().dot(delta1)#ones of the length of z2 because we differentiate each element of z2 w.r.t. b2- each element is thereofore 1
    assert  b2.shape == gradb2.shape, "b2 grad mismatch"
    h_grad = sigmoid_grad(hidden)
    gradW1 = data.transpose().dot(delta1.dot(W2.transpose()) * h_grad)
    assert  W1.shape == gradW1.shape, "W1 grad mismatch"
    gradb1 = np.ones((z1.shape[0],1)).transpose().dot(delta1.dot(W2.transpose()) * h_grad)#ones of length of z1 because each element of z1 when differentiated with b1 yields 1
    assert  b1.shape == gradb1.shape, "b1 grad mismatch"
    ### END YOUR CODE
    
    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(), 
        gradW2.flatten(), gradb2.flatten()))
    
    return cost, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using 
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1
    
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks(): 
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py 
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
