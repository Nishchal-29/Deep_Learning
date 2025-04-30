import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

# Disable oneDNN custom ops warning
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_coffee_data():
    """Creates a coffee roasting dataset.
    roasting duration: 12-15 minutes is best
    temperature range: 175-260C is best
    """
    rng = np.random.default_rng(2)
    X = rng.random(400).reshape(-1, 2)
    X[:, 1] = X[:, 1] * 4 + 11.5        # 12-15 min is best
    X[:, 0] = X[:, 0] * (285 - 150) + 150  # 175-260 C is best
    Y = np.zeros(len(X))

    # Label the data based on roasting conditions
    for i, (t, d) in enumerate(X):
        y = -3 / (260 - 175) * t + 21
        if (175 < t < 260 and 12 < d < 15 and d <= y):
            Y[i] = 1
        else:
            Y[i] = 0

    return X, Y.reshape(-1, 1)

# Load and normalize data
X, Y = load_coffee_data()

norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)
x = norm_l(X)

# Define the model
model = Sequential([
    Dense(3, activation="relu", input_shape=(2,)),
    Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=Adam(learning_rate=0.01) 
)

# Train the model
model.fit(x, Y, epochs=50)

# Test the model
x_test = np.array([[200, 13.9], [200, 17]])
x_test_n = norm_l(x_test)
predictions = model.predict(x_test_n)

# Convert probabilities to binary predictions
y_hat = tf.round(predictions)

print(f"Predictions (probabilities):\n {predictions}")
print(f"Binary classification results:\n {y_hat.numpy()}")
