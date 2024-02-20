import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

df = pd.read_excel("tabla_para_probar.xlsx")
df = df.replace('?', np.nan)

inputs = df[['x1', 'x2', 'x3', 'x4']].values
outputs = df[['d1', 'd2']].fillna(0).values.astype(int)

# Randomly select a portion of the rows to be included in the training set
train_mask = np.zeros(len(inputs), dtype=bool)
train_mask[np.random.choice(np.arange(len(inputs)), size=int(len(inputs) * 0.7), replace=False)] = True

# Split the data into training and testing sets
X_train = inputs[train_mask]
y_train = outputs[train_mask]
X_test = inputs[~train_mask]
y_test = outputs[~train_mask]
num_test = y_test.shape[0]

# Initialize weights
L = 4  # number of hidden neurons
N = 4  # number of inputs
M = 2  # number of outputs
w_h = np.random.rand(L, N) * 0.01
w_o = np.random.rand(M, L) * 0.01

# Initialize other variables
alpha = 0.1
Q = X_train.shape[0]
E = float("inf")

# Initialize lists for tracking values during training
epochs = []
losses = []
weights_h = []
weights_o = []
gradients_h = []
gradients_o = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # with a = 1

# Train model
for epoch in range(10000):
    # Reset values for this epoch
    E = 0
    DELTAw_h = np.zeros_like(w_h)
    DELTAw_o = np.zeros_like(w_o)

    # Iterate over training examples
    for j in range(Q):
        # Forward pass
        net_h = np.dot(w_h, X_train[j].T)  # Compute the net input to the hidden layer
        y_h = sigmoid(net_h)  # Apply the activation function to get the output of the hidden layer
        net_o = np.dot(w_o, y_h)  # Compute the net input to the output layer
        y = sigmoid(net_o)  # Apply the activation function to get the output of the output layer

        # Backward pass
        delta_o = (y_train[j] - y) * y * (1 - y)  # Calculate the error derivative for the output layer
        delta_h = y_h * (1 - y_h) * np.dot(w_o.T, delta_o)  # Calculate the error derivative for the hidden layer
        DELTAw_o += alpha * np.outer(delta_o, y_h)  # Compute the weight updates for the output layer
        DELTAw_h += alpha * np.outer(delta_h, X_train[j])  # Compute the weight updates for the hidden layer

        # Calculate error
        E = np.linalg.norm(delta_o)  # Accumulate the error for this example

    # Update weights
    w_o += DELTAw_o / Q  # Update the weights of the output layer
    w_h += DELTAw_h / Q  # Update the weights of the hidden layer

    # Calculate average error
    E /= Q  # Compute the average error over all examples

    # Append values to lists for tracking
    epochs.append(epoch)
    losses.append(E)
    weights_h.append(w_h.copy())
    weights_o.append(w_o.copy())
    gradients_h.append(DELTAw_h / Q)
    gradients_o.append(DELTAw_o / Q)

    # Print progress every 10 epochs
    if epoch % 100 == 0:
        print("Epoch %d: Loss = %.4f" % (epoch, E))

# Plot loss curve
plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Plot weights and gradients over time
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
axes = axes.ravel()

for i, title in enumerate(['Weights_h', 'Gradients_h', 'Weights_o', 'Gradients_o']):
    values = eval(title.lower())
    data = np.array(values)
    for j in range(data.shape[1]):
        for k in range(data.shape[2]):
            axes[i].plot(data[:, j, k], label=f'{j+1}, {k+1}')

    axes[i].set_xlabel('Epoch')
    axes[i].set_ylabel('Value')
    axes[i].set_title(title)
    axes[i].legend()

plt.tight_layout()
plt.show()

# Evaluate accuracy
y_test_pred = []
for i in range(num_test):
    net_h = np.dot(w_h, X_test[i].T)
    y_h = sigmoid(net_h)
    net_o = np.dot(w_o, y_h)
    y_test_pred.append(sigmoid(net_o))

y_test_pred = np.array(y_test_pred)

y_test_pred_bin = np.round(y_test_pred).astype(int)
y_test_bin = y_test.astype(int)

y_test_pred_flat = y_test_pred_bin.ravel()
y_test_flat = y_test_bin.ravel()

correct = np.sum(np.all(y_test_pred_bin == y_test_bin, axis=0))
accuracy = correct / num_test * 100

if num_test > 0:
    correct = np.sum(y_test_pred_bin == y_test_bin)
    accuracy = correct / (2*num_test) * 100
    print("Accuracy: %.2f%%" % accuracy)
else:
    print("No test examples with complete output data.")