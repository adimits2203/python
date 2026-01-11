# python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def _sigmoid(self, z):
        # numerically stable sigmoid
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for _ in range(self.epochs):
            linear = np.dot(X, self.w) + self.b
            preds = self._sigmoid(linear)

            # gradients (binary cross-entropy)
            dw = (1 / n_samples) * np.dot(X.T, (preds - y))
            db = (1 / n_samples) * np.sum(preds - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict_proba(self, X):
        return self._sigmoid(np.dot(X, self.w) + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
