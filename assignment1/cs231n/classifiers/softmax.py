import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  S = np.zeros([N, C])
  for i in range(N):
    for j in range(C):
      S[i][j] += np.sum(X[i] * W[:, j])
    loss += -S[i][y[i]] + np.log(np.sum(np.e ** S[i]))

  for i in range(N):
    dW.T[y[i]] += -X[i]
    for j in range(C):
      dW.T[j] += X[i] * np.e ** S[i][j] / np.sum(np.e ** S[i])


  loss /= N
  loss += 0.5 * reg * np.sum(W * W)

  dW /= N
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  S_before = X.dot(W)
  S = np.e ** S_before

  loss += -np.log(S[range(N), y] / np.sum(S, axis=1)).mean()
  loss += 0.5 * reg * np.sum(W * W)

  counts = np.zeros([N, C])
  counts[range(N), y] += -1
  counts += S / np.sum(S, axis=1).reshape(-1, 1)
  dW += X.T.dot(counts)


  dW /= N
  dW += reg * W


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

