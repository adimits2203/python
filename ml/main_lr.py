# python
from ml.regression_linear import LinearRegression
import numpy as np

def main():
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 2, size=(100, 1))
    true_b = 4.0
    true_w = np.array([3.0])
    y = (X @ true_w).flatten() + true_b + rng.normal(0, 0.5, size=100)

    model = LinearRegression(lr=0.05, epochs=2000)
    model.fit(X, y)

    y_pred = model.predict(X)
    mse = np.mean((y - y_pred) ** 2)

    print("b:", model.b)
    print("w:", model.w)
    print("MSE:", mse)

if __name__ == "__main__":
    main()