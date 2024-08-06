import numpy as np
from collections import defaultdict

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature  # 分裂特征索引
        self.threshold = threshold  # 分裂阈值
        self.left = left  # 左子树
        self.right = right  # 右子树
        self.value = value  # 叶子节点的值

class DecisionTree:
    def __init__(self, criterion='gini', min_samples_split=2):
        self.root = None
        self.criterion = criterion
        self.min_samples_split = min_samples_split

    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        impurity = 1.0
        for count in counts:
            ratio = count / len(y)
            impurity -= ratio ** 2
        return impurity

    def entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        total_samples = len(y)
        entropy_value = 0.0
        for count in counts:
            ratio = count / total_samples
            if ratio > 0:
                entropy_value -= ratio * np.log2(ratio)
        return entropy_value

    def information_gain(self, y, y_left, y_right):
        # 计算信息增益
        if self.criterion == 'gini':
            return self.gini_impurity(y) - (
                (len(y_left) / len(y)) * self.gini_impurity(y_left) +
                (len(y_right) / len(y)) * self.gini_impurity(y_right)
            )
        elif self.criterion == 'entropy':
            return self.entropy(y) - (
                (len(y_left) / len(y)) * self.entropy(y_left) +
                (len(y_right) / len(y)) * self.entropy(y_right)
            )
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'.")

    def best_split(self, X, y):
        best_gain = -1
        best_feature = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            feature_values = X[:, feature_index]
            unique_values = np.unique(feature_values)

            for threshold in unique_values:
                left_indices = feature_values <= threshold
                right_indices = feature_values > threshold

                if np.sum(left_indices) < self.min_samples_split or np.sum(right_indices) < self.min_samples_split:
                    continue

                gain = self.information_gain(y, y[left_indices], y[right_indices])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold

        return best_feature, best_threshold

    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(value=y[0])

        if len(y) < self.min_samples_split:
            return Node(value=np.bincount(y).argmax())

        feature, threshold = self.best_split(X, y)

        if feature is None:
            return Node(value=np.bincount(y).argmax())

        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold

        left_subtree = self.build_tree(X[left_indices], y[left_indices])
        right_subtree = self.build_tree(X[right_indices], y[right_indices])

        return Node(feature=feature, threshold=threshold, left=left_subtree, right=right_subtree)

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self.predict_one(node.left, x)
        else:
            return self.predict_one(node.right, x)

    def predict(self, X):
        return np.array([self.predict_one(self.root, x) for x in X])


X = np.array([[2.5, 2.4],
              [1.5, 1.8],
              [3.5, 3.6],
              [3.0, 3.2],
              [1.0, 0.6]])
y = np.array([0, 0, 1, 1, 0])

# 创建决策树实例并训练，使用基尼标准
tree_gini = DecisionTree(criterion='gini', min_samples_split=2)
tree_gini.fit(X, y)
predictions_gini = tree_gini.predict(X)
print("Gini Predictions:", predictions_gini)

# 创建决策树实例并训练，使用信息增益标准（C4.5）
tree_entropy = DecisionTree(criterion='entropy', min_samples_split=2)
tree_entropy.fit(X, y)
predictions_entropy = tree_entropy.predict(X)
print("Entropy Predictions:", predictions_entropy)