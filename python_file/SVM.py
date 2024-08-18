#/usr/bin/python3


import numpy as np
from cvxopt import matrix, solvers


class SVM:
    def __init__(self, C=1.0, kernel='linear', gamma=1.0):
        self.C = C  # normalization
        self.kernel = kernel # kernal class
        self.gamma = gamma # arugument for kernal
        self.alpha = None # lagrange multiplier
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        
    
    def kernel_function(self, X1, X2):
        if self.kernel == "linear":
            return np.dot(X1, X2.T)
        elif self.kernel == "RBF":
            sq_dists = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * sq_dists)
        else:
            return ValueError("Unsupported kernel type chose")
        
    
    def decision_function(self, X):
        K = self.kernel_function(X, self.support_vectors)
        return np.dot(K, self.alpha * self.support_vector_labels) + self.b
    
    
    def fit(self, X, y):
        n_samples, n_features = X.shape #two dimensional data
        K = self.kernel_function(X, X)
        
        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n_samples))
        G = matrix(np.vstack((-np.eye(n_samples), np.eye(n_samples))))
        h = matrix(np.hstack((np.zeros(n_samples), np.ones(n_samples) * self.C)))
        A = matrix(y, (1, n_samples), 'd')
        b = matrix(0.0)
        
        solution = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(solution['x'])
        
        # choose the support vector
        sv = self.alpha > 1e-5
        self.alpha = self.alpha[sv]
        self.support_vectors = X[sv]
        self.support_vector_labels = y[sv]
        
        self.b = np.mean(
            [self.support_vector_labels[i] - np.sum(self.alpha * self.support_vector_labels * K[sv][:, i]) 
             for i in range(len(self.alpha))])
        
        
    def predict(self, X):
        return np.sign(self.decision_function(X))
        
        
if __name__ == "__main__":
    # 生成示例数据
    X = np.array([[1, 2], [2, 3], [3, 3], [5, 1], [6, 2], [7, 3]])
    y = np.array([-1, -1, -1, 1, 1, 1])  # 类别标签

    # 创建 SVM 实例并训练
    svm = SVM(C=1.0, kernel='RBF', gamma=0.5)
    svm.fit(X, y)

    # 预测
    predictions = svm.predict(X)
    print("Predictions:", predictions)
