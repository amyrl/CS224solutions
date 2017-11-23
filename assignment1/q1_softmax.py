import numpy as np
import random

def softmax(x):
    """
    Compute the softmax function for each row of the input x.

    It is crucial that this function is optimized for speed because
    it will be used frequently in later code.
    You might find numpy functions np.exp, np.sum, np.reshape,
    np.max, and numpy broadcasting useful for this task. (numpy
    broadcasting documentation:
    http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)

    You should also make sure that your code works for one
    dimensional inputs (treat the vector as a row), you might find
    it helpful for your later problems.

    You must implement the optimization in problem 1(a) of the 
    written assignment!
    """
    shape = x.shape
    if len(shape)==1:
        constant_factor = x.max() 
        x += constant_factor
        numerator = np.exp(x)
        y = numerator/np.sum(numerator) 
        return y
    else:
        x = np.transpose(np.transpose(x) - x.max(axis=1))
        numerator = np.exp(x)
        y = numerator/numerator.sum(axis=1)[:,None]
        return y

def test_softmax_basic():
    """
    Some simple tests to get you started. 
    Warning: these are not exhaustive.
    """
    print "Running basic tests..."
    test1 = softmax(np.array([1,2]))
    print test1
    assert np.amax(np.fabs(test1 - np.array(
        [0.26894142,  0.73105858]))) <= 1e-6

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print test2
    assert np.amax(np.fabs(test2 - np.array(
        [[0.26894142, 0.73105858], [0.26894142, 0.73105858]]))) <= 1e-6

    test3 = softmax(np.array([[-1001,-1002]]))
    print test3
    assert np.amax(np.fabs(test3 - np.array(
        [0.73105858, 0.26894142]))) <= 1e-6

    print "You should verify these results!\n"

def test_softmax():
    """ 
    Use this space to test your softmax implementation by running:
        python q1_softmax.py 
    This function will not be called by the autograder, nor will
    your tests be graded.
    """
    print "Running your tests..."
    ### YOUR CODE HERE
    #test1 = softmax(np.array([3, 4]))
    #print test1

    test2 = softmax(np.array([[  6.73929562e-03,   1.41523743e-02,   6.66024598e-01,
          9.65618341e-02,   1.82109543e-01,   2.28411926e-03,
          7.64420732e-01,   1.66452746e-02,   4.77790849e-02,
          1.00000000e+00],
       [  4.07410862e-02,   8.42439977e-02,   6.46930827e-01,
          6.09260264e-02,   2.42134090e-01,   5.32951002e-03,
          2.07077367e-02,   1.05960638e-02,   1.38955369e-02,
          1.00000000e+00]]))
    print test2
    ### END YOUR CODE  

if __name__ == "__main__":
    test_softmax_basic()
    test_softmax()
