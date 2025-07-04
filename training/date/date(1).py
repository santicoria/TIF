import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist

# Load and preprocess data
csv_path = '' #Enter the earthquake dataset path here
earthquakes = pd.read_csv(csv_path)
# Filter to the region of interest
earthquakes = earthquakes[
    (earthquakes['latitude'].between(-55, -18)) &
    (earthquakes['longitude'].between(-75, -53))
].copy()

# Sort by time to ensure correct time differences
earthquakes = earthquakes.sort_values('time')

# Feature engineering with datetime handling
earthquakes['time'] = pd.to_datetime(earthquakes['time']).dt.tz_localize(None)
earthquakes = earthquakes.dropna(subset=['time'])
print("Converted 'time' column dtype:", earthquakes['time'].dtype)

earthquakes['year'] = earthquakes['time'].dt.year
earthquakes['month'] = earthquakes['time'].dt.month
earthquakes['day'] = earthquakes['time'].dt.day
reference_date = pd.Timestamp('1970-01-01')
earthquakes['time_numeric'] = (earthquakes['time'] - reference_date).dt.total_seconds() / (24 * 3600)
earthquakes['lat_lon_interaction'] = earthquakes['latitude'] * earthquakes['longitude']

# Calculate time_to_next (forward difference)
earthquakes['time_to_next'] = earthquakes['time_numeric'].diff().shift(1)
# Clip negative values and drop NaN from the first row
earthquakes = earthquakes.dropna(subset=['time_to_next'])
earthquakes['time_to_next'] = earthquakes['time_to_next'].clip(lower=0)

# Categorical encoding
categorical_features = ['magType', 'net', 'type', 'status', 'locationSource', 'magSource']
earthquakes = pd.get_dummies(earthquakes, columns=categorical_features)

# Define features
numeric_features = ['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms',
                   'horizontalError', 'depthError', 'magError', 'magNst',
                   'year', 'month', 'day', 'lat_lon_interaction']
features = numeric_features + [col for col in earthquakes.columns if col.startswith(('magType_', 'net_', 'type_', 'status_', 'locationSource_', 'magSource_'))]
X = earthquakes[features].fillna(0)
y = earthquakes['time_to_next']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences based on 200 km radius
sequence_length = 10
def create_local_sequences(X, y, coords, radius_km=200):
    Xs, ys, indices = [], [], []
    coords_array = np.array(coords)
    for i in range(len(X) - sequence_length + 1):
        base_coords = coords_array[i + sequence_length - 1]  # Coordinates of the target earthquake
        # Calculate distances to all preceding points
        distances = cdist([base_coords], coords_array[:i + sequence_length - 1])[0]
        # Find indices of earthquakes within 200 km
        local_indices = np.where(distances <= radius_km / 111.0)[0]  # Approx 111 km per degree of latitude
        if len(local_indices) >= sequence_length:
            # Sort by time (index) and take the most recent sequence_length events
            local_indices = local_indices[np.argsort(local_indices)][-sequence_length:]
            Xs.append(X[local_indices])
            ys.append(y[i + sequence_length - 1])
            indices.append(i + sequence_length - 1)
        elif len(local_indices) > 0:
            # Pad with zeros if fewer than sequence_length events
            padded_indices = np.pad(local_indices, (sequence_length - len(local_indices), 0), mode='constant', constant_values=0)
            Xs.append(X[padded_indices])
            ys.append(y[i + sequence_length - 1])
            indices.append(i + sequence_length - 1)
    return np.array(Xs), np.array(ys), np.array(indices)

# Extract coordinates
coords = earthquakes[['latitude', 'longitude']].values
X_seq, y_seq, seq_indices = create_local_sequences(X_scaled, y.values, coords)
print(f"Number of sequences created: {len(X_seq)}")  # Debug print

# Split data
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X_seq, y_seq, seq_indices, test_size=0.2, random_state=42)
print(f"Number of test samples (predicted scenarios): {len(X_test)}")  # Debug print

# Build LSTM model
model = Sequential([
    LSTM(100, activation='relu', input_shape=(sequence_length, X_scaled.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(20, activation='relu'),
    Dense(1)  # Linear output
])
model.compile(optimizer='adam', loss='mse')

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=50, batch_size=32, callbacks=[early_stopping], verbose=1)

# Evaluate
predictions = model.predict(X_test)
predictions = np.clip(predictions, 0, None)  # Clip negative values to 0
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# Calculate accuracy as percentage within ±1 day tolerance
tolerance = 1.0  # ±1 day tolerance
within_tolerance = np.abs(y_test - predictions.flatten()) <= tolerance
accuracy = np.mean(within_tolerance) * 100
print(f"\nEvaluation Metrics (time difference in days):")
print(f"MAE: {mae:.4f} days")
print(f"RMSE: {rmse:.4f} days")
print(f"R²: {r2:.4f}")
print(f"Accuracy (±{tolerance} day): {accuracy:.2f}%")

# Analyze predicted values range and density
predictions_flat = predictions.flatten()
pred_min = predictions_flat.min()
pred_max = predictions_flat.max()
print(f"\nPredicted Time Difference Range:")
print(f"Minimum: {pred_min:.4f} days")
print(f"Maximum: {pred_max:.4f} days")

# Generate CSV with predictions and actual values
results = pd.DataFrame({
    'earthquake_id': earthquakes.iloc[idx_test]['id'].values,
    'actual_time': earthquakes.iloc[idx_test]['time'].values,
    'predicted_time_to_next': predictions.flatten(),
    'actual_time_to_next': y_test,
    'expected_next_time': earthquakes.iloc[idx_test]['time'].values + pd.to_timedelta(predictions.flatten(), unit='D'),
    'actual_next_time': earthquakes.iloc[idx_test + 1]['time'].values if len(idx_test) == len(earthquakes.iloc[idx_test + 1]) else np.nan
})
results.to_csv('predictions_with_actual.csv', index=False)
print("Predictions saved to 'predictions_with_actual.csv'")

# Density plot of predicted values
plt.figure(figsize=(8, 6))
sns.kdeplot(predictions_flat, shade=True)
plt.title('Density of Predicted Time Differences (days)')
plt.xlabel('Predicted Time Difference (days)')
plt.ylabel('Density')
plt.xlim(0, pred_max + 10)  # Set x-axis limit slightly beyond max
plt.savefig('density_plot_time_diff.png')
plt.close()

# Visualization functions
def plot_scatter_plot():
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions.flatten(), alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Time Difference (days)')
    plt.ylabel('Predicted Time Difference (days)')
    plt.title('Actual vs Predicted Time Difference')
    plt.yscale('linear')
    plt.xscale('linear')
    plt.xlim(0, 30)  # Adjusted to cover max predicted value + buffer
    plt.ylim(0, 30)  # Adjusted to cover max predicted value + buffer
    plt.savefig('scatter_plot_time_diff_zoomed.png')
    plt.close()

def plot_error_distribution():
    plt.figure(figsize=(8, 6))
    errors = y_test - predictions.flatten()
    sns.histplot(errors, kde=True)
    plt.title('Error Distribution for Time Difference (days)')
    plt.xlabel('Prediction Error (days)')
    plt.savefig('error_distribution_time_diff.png')
    plt.close()

def plot_training_history():
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE in days²)')
    plt.legend()
    plt.savefig('training_history_time_diff.png')
    plt.close()

# Generate plots
plot_scatter_plot()
plot_error_distribution()
plot_training_history()

# Save model and scaler
model.save('earthquake_time_model.h5')
joblib.dump(scaler, 'scaler_time.pkl')
print("Model, scaler, and plots saved successfully.")
