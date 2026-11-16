import numpy as np

probs = np.array([[0.7, 0.2, 0.1],
                  [0.1, 0.6, 0.3],
                  [0.2, 0.2, 0.6],
                  [0.5, 0.3, 0.2],
                  [0.1, 0.3, 0.6]])
y_test = np.array([0, 1, 2, 1, 2])
w_true = np.array([150.0, 120.0, 200.0, 180.0])
w_pred = np.array([140.0, 130.0, 210.0, 170.0])

# Predicted classes
y_pred = np.argmax(probs, axis=1)
print("Predicted:", y_pred)

# Accuracy
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)

# MSE of weight predictions
mse = np.mean((w_true - w_pred) ** 2)
print("MSE:", mse)

# Wrongly predicted samples (required for the second commit)
wrong_mask = y_pred != y_test
print("Wrongly predicted samples (mask):", wrong_mask)
print("Wrong sample indices:", np.where(wrong_mask)[0])