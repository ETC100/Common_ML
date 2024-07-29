#usr/bin/python


import numpy as np

class linear_regression:
	def __init__(self, lr, iter_count):
		self.learn_rate = lr
		self.iter_count = iter_count
		self.theta = None
		self.b = None
	
	def fit(self, X, y):
		samples = len(y)
		self.b = 0
		self.theta = np.zeros(X.shape[1])
		for _ in range(self.iter_count):
			predictions = self.predict(X)
			errors = predictions - y
			self.b -= self.learn_rate * np.sum(errors) / samples
			self.theta -= self.learn_rate * X.T.dot(errors) / samples
			# theta = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
	
	def predict(self, X):
		return X.dot(self.theta) + self.b
	
	def mean_squared_error(self, y_true, y_pred):
		return np.mean((y_true - y_pred) ** 2)
	
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(0)
    X = np.random.rand(100, 3)  # 100 个样本，3 个特征
    true_coefficients = np.array([1.5, -2.0, 3.0])
    y = X @ true_coefficients + np.random.normal(0, 0.1, size=100)

    # 创建模型并训练
    model = linear_regression(lr=0.01, iter_count=1000)
    model.fit(X, y)

    # 进行预测
    y_pred = model.predict(X)

    # 计算均方误差
    mse = model.mean_squared_error(y, y_pred)
    print("偏置项 b:", model.b)
    print("回归系数:", model.theta)
    print("均方误差:", mse)