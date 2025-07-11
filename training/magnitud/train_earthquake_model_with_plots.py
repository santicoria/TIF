import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.spatial import cKDTree
import joblib
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load earthquake data
earthquakes = pd.read_csv('combined_earthquakes.csv')
faults = pd.read_csv('fault_lines.csv')

# Print available columns to debug
print("Available columns in earthquakes DataFrame:", earthquakes.columns.tolist())

# Filter data for Argentina and Chile (approx. lat: -55 to -18, lon: -75 to -53)
earthquakes = earthquakes[
    (earthquakes['latitude'].between(-55, -18)) & 
    (earthquakes['longitude'].between(-75, -53))
]

# Sample 50% of the data to include more data points
earthquakes = earthquakes.sample(frac=0.5, random_state=42)

# Calculate distance to nearest fault using cKDTree
earthquake_coords = earthquakes[['latitude', 'longitude']].values
fault_coords = faults[['latitude', 'longitude']].values
tree = cKDTree(fault_coords)
distances, _ = tree.query(earthquake_coords, k=1)
earthquakes['dist_to_fault'] = distances

# Feature engineering
earthquakes['time'] = pd.to_datetime(earthquakes['time']).dt.tz_localize(None)
earthquakes['year'] = earthquakes['time'].dt.year
earthquakes['month'] = earthquakes['time'].dt.month
earthquakes['day'] = earthquakes['time'].dt.day
reference_date = pd.Timestamp('1970-01-01')
earthquakes['time_numeric'] = (earthquakes['time'] - reference_date).dt.total_seconds() / (24 * 3600)

# Prepare features (all columns except mag, time, id, updated, place)
numeric_features = ['latitude', 'longitude', 'depth', 'nst', 'gap', 'dmin', 'rms', 
                   'horizontalError', 'depthError', 'magError', 'magNst', 'dist_to_fault',
                   'year', 'month', 'day', 'time_numeric']
categorical_features = ['magType', 'net', 'type', 'status', 'locationSource', 'magSource']

# One-hot encode categorical features
earthquakes = pd.get_dummies(earthquakes, columns=categorical_features)

# Update features list with one-hot encoded columns
features = numeric_features + [col for col in earthquakes.columns if col.startswith(('magType_', 'net_', 'type_', 'status_', 'locationSource_', 'magSource_'))]
target_column = 'mag' if 'mag' in earthquakes.columns else 'magnitude' if 'magnitude' in earthquakes.columns else None
if target_column is None:
    raise KeyError("Neither 'mag' nor 'magnitude' found in the DataFrame columns.")
X = earthquakes[features].fillna(0)  # Handle missing values
y = earthquakes[[target_column]]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build ANN model for magnitude with increased complexity and dropout
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for magnitude
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train[target_column], 
                    validation_data=(X_test_scaled, y_test[target_column]),
                    epochs=100, batch_size=16, verbose=1, callbacks=[early_stopping])

# Make predictions
predictions = model.predict(X_test_scaled).flatten()

# Evaluate model
actual = y_test[target_column]
print(f"\nEvaluation for {target_column}:")
print(f"Actual range: {actual.min():.2f} to {actual.max():.2f}")
print(f"Predicted range: {predictions.min():.2f} to {predictions.max():.2f}")
print(f"MAE: {mean_absolute_error(actual, predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual, predictions)):.4f}")
print(f"R²: {r2_score(actual, predictions):.4f}")

# Plotting functions
def plot_scatter_plot():
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test[target_column], predictions, alpha=0.5)
    plt.plot([y_test[target_column].min(), y_test[target_column].max()], 
             [y_test[target_column].min(), y_test[target_column].max()], 'r--')
    plt.xlabel(f'Actual {target_column}')
    plt.ylabel(f'Predicted {target_column}')
    plt.title(f'Actual vs Predicted {target_column}')
    plt.savefig(f'scatter_plot_{target_column}.png')
    plt.close()

def plot_error_distribution():
    plt.figure(figsize=(8, 6))
    errors = y_test[target_column] - predictions
    sns.histplot(errors, kde=True)
    plt.title(f'Error Distribution for {target_column}')
    plt.xlabel('Prediction Error')
    plt.savefig(f'error_distribution_{target_column}.png')
    plt.close()

def plot_training_history():
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Training and Validation Loss for {target_column}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.savefig(f'training_history_{target_column}.png')
    plt.close()

def plot_confusion_matrix():
    # Discretize magnitudes into bins (e.g., 0-3, 3-4, 4-5, 5-6, 6-7, 7-8)
    bins = [0, 3, 4, 5, 6, 7, 8]
    actual_binned = pd.cut(y_test[target_column], bins=bins, labels=False, include_lowest=True)
    predicted_binned = pd.cut(predictions, bins=bins, labels=False, include_lowest=True)
    cm = pd.crosstab(actual_binned, predicted_binned, rownames=['Actual'], colnames=['Predicted'], normalize='index')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
    plt.title(f'Confusion Matrix for {target_column}')
    plt.savefig(f'confusion_matrix_{target_column}.png')
    plt.close()

# Generate plots
plot_scatter_plot()
plot_error_distribution()
plot_training_history()
plot_confusion_matrix()

# Save the model and scaler
model.save(f'earthquake_{target_column}_model.h5')
joblib.dump(scaler, f'scaler_{target_column}.pkl')

print("Plots and model saved successfully.")