from ml.LogisticRegression import LogisticRegression
import numpy as np

def main():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2, size=(200, 1))
    true_b = -1.0
    true_w = np.array([3.0])

    logits = (X @ true_w).flatten() + true_b
    probs = 1 / (1 + np.exp(-logits))
    y = rng.binomial(1, probs, size=probs.shape[0])

    model = LogisticRegression(lr=0.5, epochs=2000)
    model.fit(X, y)

    y_pred = model.predict(X)
    acc = np.mean(y_pred == y)
    print("b:", model.b)
    print("w:", model.w)
    print("Accuracy:", acc)


if __name__ == "__main__":
    main()