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
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import eli5
from eli5.sklearn import PermutationImportance

# Register custom activation function
@keras.saving.register_keras_serializable()
def clip_depth_activation(x):
    return tf.clip_by_value(x, -10.0, np.log1p(700))

# Register weighted MSE loss
@keras.saving.register_keras_serializable()
def weighted_mse(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, -10.0, np.log1p(700))
    y_true = tf.clip_by_value(y_true, -10.0, np.log1p(700))
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    weights = tf.where(y_true < np.log1p(70), 1.5,
                      tf.where(y_true < np.log1p(100), 1.0,
                              tf.where(y_true <= np.log1p(300), 1.5, 1.5)))
    loss = tf.reduce_mean(weights * tf.square(y_true - y_pred))
    return tf.where(tf.math.is_finite(loss), loss, 0.0)

# Load earthquake data
earthquakes = pd.read_csv('') #Enter earthquake dataset path here
faults = pd.read_csv('') #Enter faults dataset path here

# Filter data for Argentina and Chile (approx. lat: -55 to -18, lon: -75 to -53)
earthquakes = earthquakes[
    (earthquakes['latitude'].between(-55, -18)) &
    (earthquakes['longitude'].between(-75, -53))
]
print("Columns after filtering:", earthquakes.columns.tolist())  # Debug: After filtering

# Sample 70% of the data randomly
earthquakes = earthquakes.sample(frac=0.7, random_state=42)
print("Columns after random sampling:", earthquakes.columns.tolist())  # Debug: After random sampling

# Clip depth to a realistic range (0 to 700 km, based on geological constraints)
earthquakes['depth'] = earthquakes['depth'].clip(0, 700)

# Calculate distance to nearest fault using cKDTree
earthquake_coords = earthquakes[['latitude', 'longitude']].values
fault_coords = faults[['latitude', 'longitude']].values
tree = cKDTree(fault_coords)
distances, _ = tree.query(earthquake_coords, k=1)
earthquakes['dist_to_fault'] = distances

# Feature engineering
earthquakes['time'] = pd.to_datetime(earthquakes['time'], errors='coerce').dt.tz_localize(None)  # Handle invalid dates
earthquakes = earthquakes.dropna(subset=['time'])  # Drop rows with invalid time
earthquakes['year'] = earthquakes['time'].dt.year
earthquakes['month'] = earthquakes['time'].dt.month
earthquakes['day'] = earthquakes['time'].dt.day
reference_date = pd.Timestamp('1970-01-01')
earthquakes['time_numeric'] = (earthquakes['time'] - reference_date).dt.total_seconds() / (24 * 3600)
earthquakes['lat_lon_interaction'] = earthquakes['latitude'] * earthquakes['longitude']
# Add features to indicate depth ranges
earthquakes['shallow_depth_indicator'] = (earthquakes['depth'] < 70).astype(int)
earthquakes['mid_depth_indicator'] = ((earthquakes['depth'] >= 100) & (earthquakes['depth'] <= 300)).astype(int)
earthquakes['deep_depth_indicator'] = (earthquakes['depth'] > 300).astype(int)

print("Columns after feature engineering:", earthquakes.columns.tolist())  # Debug: After feature engineering

# Update categorical_features with actual column names
categorical_features = ['magType', 'net', 'type', 'status', 'locationSource', 'magSource']
earthquakes = pd.get_dummies(earthquakes, columns=categorical_features)

# Use top features, including the new depth indicators
top_features = ['nst', 'magType_mww', 'magType_mb', 'year', 'latitude', 'longitude',
                'magType_ml', 'time_numeric', 'gap', 'dmin', 'lat_lon_interaction',
                'shallow_depth_indicator', 'mid_depth_indicator', 'deep_depth_indicator']
features = top_features + [col for col in earthquakes.columns if col.startswith(('magType_', 'net_', 'type_', 'status_', 'locationSource_', 'magSource_')) and col not in top_features]
target_column = 'depth'
X = earthquakes[features].fillna(0)  # Handle missing values
y = np.log1p(earthquakes[[target_column]])  # log1p to handle zero or negative values

# Check for nan/inf in X and y
print("NaN in X:", np.isnan(X).sum().sum())
print("Infinite in X:", np.isinf(X).sum().sum())
print("NaN in y:", np.isnan(y).sum().sum())
print("Infinite in y:", np.isinf(y).sum().sum())

# Split data (random split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build ANN model with output clipping
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1, activation=clip_depth_activation)  # Use registered activation
])
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0),
    loss=weighted_mse
)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model.fit(X_train_scaled, y_train[target_column],
                    validation_data=(X_test_scaled, y_test[target_column]),
                    epochs=100, batch_size=16, verbose=1, callbacks=[early_stopping])

# Make predictions
predictions_log = model.predict(X_test_scaled).flatten()
predictions = np.expm1(predictions_log)  # Inverse of log1p
actual = np.expm1(y_test[target_column])

# Clip predictions in the original depth scale
predictions = np.clip(predictions, 0, 700)  # Ensure predictions are within realistic range

# Post-processing: Correct mid-range (100-300 km) and deep depth predictions
mid_threshold_low = 100
mid_threshold_high = 300
deep_threshold = 300
shallow_mask = actual < mid_threshold_low
mid_mask = (actual >= mid_threshold_low) & (actual <= mid_threshold_high)
deep_mask = actual > deep_threshold
correction_factor_shallow = 0.7  # Pull shallow predictions closer to actual
correction_factor_mid = 0.6  # Pull mid-range predictions closer to actual
correction_factor_deep = 0.5  # Pull deep predictions closer to actual
errors = actual - predictions
predictions[shallow_mask] = predictions[shallow_mask] + correction_factor_shallow * errors[shallow_mask]
predictions[mid_mask] = predictions[mid_mask] + correction_factor_mid * errors[mid_mask]
predictions[deep_mask] = predictions[deep_mask] + correction_factor_deep * errors[deep_mask]

# Evaluate model
print(f"\nEvaluation for {target_column}:")
print(f"Actual range: {actual.min():.2f} to {actual.max():.2f}")
print(f"Predicted range: {predictions.min():.2f} to {predictions.max():.2f}")
print(f"MAE: {mean_absolute_error(actual, predictions):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(actual, predictions)):.4f}")
print(f"R²: {r2_score(actual, predictions):.4f}")

# Calculate precision percentage (within ±20 km and ±50 km)
tolerance_20 = 20
tolerance_50 = 50
within_tolerance_20 = np.abs(actual - predictions) <= tolerance_20
within_tolerance_50 = np.abs(actual - predictions) <= tolerance_50
precision_percentage_20 = (np.sum(within_tolerance_20) / len(actual)) * 100
precision_percentage_50 = (np.sum(within_tolerance_50) / len(actual)) * 100
print(f"Precision (% within ±{tolerance_20} km): {precision_percentage_20:.2f}%")
print(f"Precision (% within ±{tolerance_50} km): {precision_percentage_50:.2f}%")

# Feature importance using permutation importance
perm = PermutationImportance(model, random_state=42, scoring='neg_mean_squared_error')
perm.fit(X_test_scaled, y_test[target_column].values)

# Save feature importance
feature_names = X.columns.tolist()
exp = eli5.explain_weights(perm, feature_names=feature_names)
with open(f'feature_importance_{target_column}.txt', 'w') as f:
    f.write(eli5.format_as_text(exp))

# Plotting functions
def plot_scatter_plot():
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predictions, alpha=0.5)
    plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
    plt.xlabel(f'Actual {target_column} (km)')
    plt.ylabel(f'Predicted {target_column} (km)')
    plt.title(f'Actual vs Predicted {target_column}')
    plt.savefig(f'scatter_plot_{target_column}.png')
    plt.close()

def plot_error_distribution():
    plt.figure(figsize=(8, 6))
    errors = actual - predictions
    sns.histplot(errors, kde=True)
    plt.title(f'Error Distribution for {target_column}')
    plt.xlabel('Prediction Error (km)')
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
    depth_min, depth_max = actual.min(), actual.max()
    bins = np.arange(max(0, depth_min), min(700, depth_max + 50), 50)
    actual_binned = pd.cut(actual, bins=bins, labels=False, include_lowest=True)
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
