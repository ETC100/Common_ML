#usr/bin/python3


import numpy as np
from collections import defaultdict


arr_t=np.array([
            ['东','干燥','强',33,'晴天'],
            ['南','潮湿','强',31,'晴天'],
            ['南','潮湿','弱',24,'阴天'],
            ['西','干燥','强',27,'阴天'],
            ['西','潮湿','强',30,'阴天'],
            ['南','适中','弱',22,'雨天'],
            ['北','潮湿','弱',20,'雨天'],
            ['南','干燥','强',34,'晴天'],
            ['北','干燥','强',26,'阴天']
            ])

class Naive_Bayes:
    def __init__(self):
        self.classPrior = {}
        self.classes = []
        self.conditional_probs = defaultdict(lambda: defaultdict(dict))
    
    def fit(self, X, y):
        total_samples = len(y)
        self.classes = np.unique(y)
        
        for cl in self.classes:
            self.classPrior[cl] = np.sum(y == cl) / total_samples
        
        for cl in self.classes:
            subset = X[y == cl]
            for feature_index in range(X.shape[1]):
                if feature_index == 3:
                    mean = np.mean(subset[:, feature_index].astype(float))
                    variance = np.var(subset[:, feature_index].astype(float))
                    self.conditional_probs[cl][feature_index] = (mean, variance)
                else:
                    feature, counts = np.unique(subset[:, feature_index], return_counts=True)
                    total_count = np.sum(counts)
                    for value, count in zip(feature, counts):
                        self.conditional_probs[cl][feature_index][value] = count / total_count
    
    def gaussian_density(self, x, mean, variance):
        # gaussian distribution for numercial object
        if variance == 0: return 0
        exponent = np.exp(-0.5 * ((x - mean) ** 2) / variance)
        return (1 / np.sqrt(2 * np.pi * variance)) * exponent
    
    def predict(self, X):
        predictions = []
        for sample in X:
            class_probs = {}
            for cl in self.classes:
                class_prob = self.classPrior[cl]
                for feature_index in range(len(sample)):
                    feature_value = sample[feature_index]
                    if feature_index == 3:
                        mean, variance = self.conditional_probs[cl][feature_index]
                        class_prob *= self.gaussian_density(float(feature_value), mean, variance)
                    else:
                        class_prob *= self.conditional_probs[cl][feature_index].get(feature_value, 1e-6)
                    
                class_probs[cl] = class_prob
            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        return np.array(predictions)
    

if __name__ == '__main__':
    X = arr_t[:, :-1]
    y = arr_t[:, -1]
    
    classifier = Naive_Bayes()
    classifier.fit(X, y)
    
    test_data = np.array([
        ['东', '适中', '强', 28],
        ['南', '干燥', '弱', 30],
        ['北', '潮湿', '强', 22]
    ])
    
    # 进行预测
    predictions = classifier.predict(test_data)
    print("Prediction_result:", predictions)
        
        