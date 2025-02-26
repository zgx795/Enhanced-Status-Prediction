
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Load the dataset
file_path = r"C:/Users/13844/Desktop/papers/paper4/data/MSS_modified_test.csv"
data = pd.read_csv(file_path)

# Convert 'date' column to datetime and set as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Create sequences for multi-step input and output
def create_sequences(data, input_steps, output_steps):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i:i+input_steps])
        y.append(data[i+input_steps:i+input_steps+output_steps])
    return np.array(X), np.array(y)

# Parameters
input_steps = 48   # Number of historical data points used for prediction
output_steps = 12  # Number of future data points to predict

# Create sequences
X, y = create_sequences(normalized_data, input_steps, output_steps)

# Split the data into train, validation, and test sets
train_size = int(len(X) * 0.8)
val_size = int(len(X) * 0.1)

X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

# Build the model
model = Sequential([
    LSTM(128, activation='tanh', input_shape=(input_steps, X.shape[2]), return_sequences=True),
    Dropout(0.2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(output_steps * X.shape[2]),
    Reshape((output_steps, X.shape[2]))
])

# Compile the model
model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=['mse'])

# Output directory for model and results
output_dir = r'C:/Users/13844/Desktop/papers/paper4/results/LSTM482412'
os.makedirs(output_dir, exist_ok=True)

# Define model save path
model_save_path = os.path.join(output_dir, 'best_model.h5')

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath=model_save_path, monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint]
)

# Plot the training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

# Load the best model
best_model = load_model(model_save_path)

# Predict on test data using the best model
y_pred = best_model.predict(X_test)

# Inverse transform predictions and actual values
y_pred_rescaled = scaler.inverse_transform(y_pred.reshape(-1, y_pred.shape[2]))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, y_test.shape[2]))

# Define SMAPE metric
def smape(y_true, y_pred):
    return np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-10)) * 100

# Calculate SMAPE
smape_score = smape(y_test_rescaled, y_pred_rescaled)
print(f"SMAPE: {smape_score:.4f}%")

# Feature information
feature_info = [
    ("MFBT", "°C"), ("MRBT", "°C"), ("CUR", "A"), ("OP", "kPa"), ("FSBT", "°C"),
    ("VBF", "%"), ("RSBT", "°C"), ("TBT", "°C"), ("PACT", "°C"), ("PBCT", "°C"), ("PCCT", "°C")
]

# Metrics storage
metrics = []

# Performance metrics and visualization
rows, cols = 4, 3
fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
fig.suptitle('GroundTruth vs Prediction for All Features at Step 0', fontsize=16, weight='bold')

for feature_idx, (feature_name, unit) in enumerate(feature_info):
    row, col = divmod(feature_idx, cols)
    ax = axes[row, col]

    # Extract true and predicted values for the feature
    true_values = y_test[:, 11, feature_idx]
    pred_values = y_pred[:, 11, feature_idx]

    # Plot comparison
    ax.plot(true_values, label='GroundTruth', linestyle='--', linewidth=1)
    ax.plot(pred_values, label='Prediction', linestyle='-', linewidth=1)
    
    ax.set_title(f"{feature_name}", fontsize=12)
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel(f"Value ({unit})", fontsize=10)
    ax.legend(fontsize=8, loc='upper right', frameon=False)
    ax.tick_params(axis='both', labelsize=8)

    # Compute metrics
    mae = np.mean(np.abs(true_values - pred_values))
    mse = np.mean((true_values - pred_values) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_values - pred_values) / true_values)) * 100
    mspe = np.mean(((true_values - pred_values) / true_values) ** 2) * 100
    r2 = 1 - np.sum((true_values - pred_values) ** 2) / np.sum((true_values - np.mean(true_values)) ** 2)

    # Append metrics
    metrics.append({
        'Feature': feature_name,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE (%)': mape,
        'MSPE (%)': mspe,
        'R²': r2
    })
    
    print(f"Feature: {feature_name}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}%, MSPE: {mspe:.4f}%, R²: {r2:.4f}")

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save the plot
plot_path = os.path.join(output_dir, 'Feature_Step_Comparison.png')
plt.savefig(plot_path, dpi=300)
plt.show()

# Save performance metrics to CSV
metrics_path = os.path.join(output_dir, 'Performance_Metrics.csv')
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv(metrics_path, index=False)

# Save ground truth and prediction data to CSV
csv_path_true = os.path.join(output_dir, 'GroundTruth_Data.csv')
csv_path_pred = os.path.join(output_dir, 'Prediction_Data.csv')
pd.DataFrame(y_test_rescaled).to_csv(csv_path_true, index=False)
pd.DataFrame(y_pred_rescaled).to_csv(csv_path_pred, index=False)

print(f"Plot saved at: {plot_path}")
print(f"GroundTruth data saved at: {csv_path_true}")
print(f"Prediction data saved at: {csv_path_pred}")
print(f"Performance metrics saved at: {metrics_path}")
