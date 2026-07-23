import numpy as np
X = np.array([[0.5, 1.2, -0.3, 0.8],
              [1.0, -0.5, 0.7, 0.2],
              [-0.6, 0.9, 1.1, -0.4]])

W_Q = np.array([[0.2, 0.1],
                [0.0, 0.3],
                [0.4, -0.2],
                [0.1, 0.5]])
W_K = np.array([[0.3, 0.2],
                [0.1, 0.4],
                [-0.2, 0.1],
                [0.5, -0.3]])
W_V = np.array([[0.1, 0.0, 0.2, 0.5],
                [0.2, 0.3, -0.1, 0.11],
                [0.4, -0.1, 0.0, 0.47],
                [0.0, 0.5, 0.3, 0.21]])

# create query, key and value matrices
Q = np.dot(X, W_Q)  # shape (3, 2)
K = np.dot(X, W_K)  # shape (3, 2)
V = np.dot(X, W_V)  # shape (3, 4)

# display it nicely formatted
np.set_printoptions(precision=4, suppress=True)
print("Query matrix Q:\n", Q)
print("Key matrix K:\n", K)
print("Value matrix V:\n", V)

d_k = Q.shape[1]  # dimension of the key vectors
scores = np.dot(Q, K.T) / np.sqrt(d_k)  # shape (3, 3)


# apply softmax to each row
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


attention_weights = softmax(scores)  # shape (3, 3)
output = np.dot(attention_weights, V)  # shape (3, 4)


