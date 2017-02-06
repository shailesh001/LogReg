import numpy as np

# N - Number, D - Dimensions, T - Target
# X - Data Matrix, Y - Output of Logistic Regression
# Cost Function = Error Function = Objective Function

N = 100
D = 2

X = np.random.randn(N,D)

# Create 2 Gaussian Clouds at (-2,-2) and (+2,+2)
# Set the first 50 items at (-2, -2)
X[:50, :] = X[:50,:] - 2*np.ones((50,D))

# Set last 50 items at (+2,+2)
X[50:, :] = X[50:, :] + 2*np.ones((50,D))

# labels: first 50 are 0, last 50 are 1
T = np.array([0]*50 + [1]*50)

# Add a column of ones
ones = np.array([[1]*N]).T
Xb = np.concatenate((ones, X), axis=1)

# randomly initialise the weights
w = np.random.randn(D + 1)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

Y = sigmoid(z)

def cross_entropy(T, Y):
    E = 0
    for i in range(N):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E

print(cross_entropy(T, Y))

learning_rate = 0.1
for i in range(200):
    if i % 10 == 0:
        print(cross_entropy(T, Y))

    w += learning_rate * np.dot((T - Y).T, Xb)
    Y = sigmoid(Xb.dot(w))

print('Final w:', w)