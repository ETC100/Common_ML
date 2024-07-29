#usr/bin/python


import numpy as np


class logistic_regression:
	def __init__(self, lr, iter_count):
		self.lr = lr
		self.iter_count = iter_count
		self.theta = None
		self.b = 0
	
	def fit(self, X, y):
		self.b = 0
		self.theta = np.zeros(X.shape[1])
		sample = len(y)
		for _ in range(self.iter_count):
			linear_model = X.dot(self.theta)
			predictions = self.sigmoid(linear_model)
			errors = predictions - y
			self.theta -= self.lr * (X.T @ errors) / sample
			self.b -= self.lr * np.sum(errors) / sample
		
	def sigmoid(self, x):
		return 1.0/(1+np.exp(-x))
	
	def predict(self, X):
		linear_model = X.dot(self.theta)
		predictions = self.sigmoid(linear_model)
		return np.where(predictions >= 0.5, 1, 0)
	
	def accuracy(self, y_true, y_pred):
		return np.mean(y_true == y_pred)
	

if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)

    # 创建模型并训练
    model = logistic_regression(lr=0.1, iter_count=1000)
    model.fit(X, y)

    # 进行预测
    y_pred = model.predict(X)

    # 计算准确率
    accuracy = model.accuracy(y, y_pred)
    print("权重:", model.theta)
    print("准确率:", accuracy)

		
		
		
		
		
		
		
		
		
		