#/usr/bin/python3


import numpy as np


class Perceptron:
	def __init__(self, lr=0.01, n_iterations=1000):
		self.lr = lr
		self.n_iterations = n_iterations
		self.theta = None
		self.b = 0
		
	def sign(self, x):
		return np.dot(x, self.theta) + self.b
	
	def fit(self, X, y):
		# 0 dimension for sample size, 1 dimension for feature count
		samples, features = X.shape[0], X.shape[1] 
		self.theta = np.zeros(features)
		
		for _ in range(self.n_iterations):
			wrong_point = 0
			for num in range(samples):
				point_X = X[num]
				point_y = y[num]
				if self.sign(point_X) * point_y <= 0:
					self.theta += self.lr * point_y * point_X
					self.b += self.lr * point_y
					wrong_point += 1
			if wrong_point == 0:
				break

	def predict(self, X):
		return np.where(self.sign(X) >= 0, 1, -1)

if __name__ == "__main__":
    # generate data
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = np.where(X[:, 0] + X[:, 1] > 1, 1, -1)

    model = Perceptron(lr=0.1, n_iterations=1000)
    model.fit(X, y)

    predictions = model.predict(X)

    accuracy = np.mean(predictions == y)
    print("Weight:", model.theta)
    print("Bias:", model.b)
    print("Accuracy:", accuracy)

