# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import time
import keras
import os
from scipy.spatial import cKDTree
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify

# Register custom activation and loss functions
@keras.saving.register_keras_serializable(package="Custom", name="clip_depth_activation")
def clip_depth_activation(x):
    return tf.clip_by_value(x, -10.0, np.log1p(700))

@keras.saving.register_keras_serializable(package="Custom", name="weighted_mse")
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

# Load all models and scalers with error handling
try:
    latitude_model = keras.models.load_model('earthquake_latitude_model.h5', compile=False)
    longitude_model = keras.models.load_model('earthquake_longitude_model.h5', compile=False)
    depth_model = keras.models.load_model('earthquake_depth_model.h5', compile=False)
    magnitude_model = keras.models.load_model('earthquake_mag_model.h5', compile=False)
    time_model = keras.models.load_model('earthquake_time_model.h5', compile=False)
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

try:
    scaler_latitude = joblib.load('scaler_latitude.pkl')
    scaler_longitude = joblib.load('scaler_longitude.pkl')
    scaler_depth = joblib.load('scaler_depth.pkl')
    scaler_magnitude = joblib.load('scaler_mag.pkl')
    scaler_time = joblib.load('scaler_time.pkl')
    print("All scalers loaded successfully.")
    lat_expected_features = scaler_latitude.feature_names_in_
    lon_expected_features = scaler_longitude.feature_names_in_
    depth_expected_features = scaler_depth.feature_names_in_
    mag_expected_features = scaler_magnitude.feature_names_in_
    time_expected_features = scaler_time.feature_names_in_
except Exception as e:
    print(f"Error loading scalers: {e}")
    exit()

# Define the trained region coordinates
MIN_LAT, MAX_LAT = -55, -18
MIN_LON, MAX_LON = -75, -53

# Feature sets based on training setup
categorical_features = ['magType', 'net', 'type', 'status', 'locationSource', 'magSource']
lat_top_features = ['nst', 'year', 'longitude', 'time_numeric', 'gap', 'dmin', 'lat_lon_interaction']
lon_top_features = ['nst', 'year', 'latitude', 'time_numeric', 'gap', 'dmin', 'lat_lon_interaction']
depth_top_features = ['nst', 'year', 'latitude', 'longitude', 'time_numeric', 'gap', 'dmin', 'lat_lon_interaction',
                     'shallow_depth_indicator', 'mid_depth_indicator', 'deep_depth_indicator']

# CSV files
API_CSV_FILE = 'api_earthquakes.csv'
PREDICTIONS_CSV_FILE = 'earthquake_predictions.csv'

# Initialize the CSV files if they don't exist
try:
    predictions_df = pd.read_csv(PREDICTIONS_CSV_FILE)
except FileNotFoundError:
    predictions_df = pd.DataFrame(columns=['earthquake_id', 'latitude', 'longitude', 'depth', 'predicted_latitude',
                                          'predicted_longitude', 'predicted_depth', 'predicted_magnitude',
                                          'predicted_time', 'prediction_timestamp', 'predicted_earthquake_id',
                                          'prediction_correct'])
    predictions_df.to_csv(PREDICTIONS_CSV_FILE, index=False)

# Initialize Flask app for web server
app = Flask(__name__, template_folder='templates')
predicted_earthquakes = []
api_earthquakes = []

def compute_time_features(df, new_event_time):
    coords = df[['latitude', 'longitude']].values
    times = df['time_numeric'].values
    tree = cKDTree(coords)
    time_since_last = []
    for i, (lat, lon, t) in enumerate(zip(coords[:, 0], coords[:, 1], times)):
        dists, idxs = tree.query([lat, lon], k=len(coords))
        mask = (dists > 0) & (dists <= 0.1) & (times[idxs] < t)
        if np.any(mask):
            prior_times = times[idxs[mask]]
            last_time = np.max(prior_times)
            time_since_last.append(t - last_time)
        else:
            time_since_last.append(0)
    df['time_since_last'] = time_since_last
    return df

def fetch_earthquake_data():
    api_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    params = {
        "format": "geojson",
        "starttime": "2025-06-01",
        "endtime": current_time,
        "minmagnitude": 2.5,
        "latitude": "-36.5",
        "longitude": "-64",
        "maxradiuskm": 2000,
        "limit": 20000
    }
    try:
        response = requests.get(api_url, params=params)
        if response.status_code == 200:
            data = response.json()
            print(f"Fetched {len(data['features'])} earthquake features")
            return data['features']
        else:
            print(f"Error fetching data: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching USGS data: {e}")
        return []

def fetch_and_save_api_data():
    global api_earthquakes
    api_url = "https://earthquake.usgs.gov/fdsnws/event/1/query"
    params = {
        "format": "geojson",
        "starttime": "2025-06-01",
        "endtime": "2025-06-17T22:00:00",  # Updated to current time
        "minmagnitude": 2.5,
        "latitude": "-36.5",
        "longitude": "-64",
        "maxradiuskm": 2000,  # Increased to capture more data
        "limit": 20000
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        api_earthquakes = []
        for feature in data['features']:
            props = feature['properties']
            geom = feature['geometry']
            quake = {
                'id': feature['id'],  # Add earthquake ID
                'time': pd.to_datetime(props['time'], unit='ms').isoformat(),
                'latitude': geom['coordinates'][1],
                'longitude': geom['coordinates'][0],
                'depth': geom['coordinates'][2] if len(geom['coordinates']) > 2 else 0,
                'magnitude': props['mag'] if 'mag' in props else 0,
                'magType': props.get('magType', 'unknown'),
                'net': props.get('net', 'unknown'),
                'type': props.get('type', 'unknown'),
                'status': props.get('status', 'unknown'),
                'locationSource': props.get('locationSource', 'unknown'),
                'magSource': props.get('magSource', 'unknown')
            }
            api_earthquakes.append(quake)
        api_df = pd.DataFrame(api_earthquakes)
        if os.path.exists(API_CSV_FILE):
            existing_df = pd.read_csv(API_CSV_FILE)
            api_df = pd.concat([existing_df, api_df]).drop_duplicates(subset=['id', 'time', 'latitude', 'longitude']).reset_index(drop=True)
        api_df.to_csv(API_CSV_FILE, index=False)
        print(f"Saved {len(api_df)} earthquakes to {API_CSV_FILE}")
    else:
        print(f"Failed to fetch API data: {response.status_code}")

def process_earthquake_data_for_latitude(current_df, earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df):
    try:
        current_df['year'] = event_time.year
        current_df['month'] = event_time.month
        current_df['day'] = event_time.day
        current_df['lat_lon_interaction'] = current_df['latitude'] * current_df['longitude']
        if not historical_df.empty:
            combined_df = pd.concat([historical_df, current_df]).reset_index(drop=True)
            combined_df = compute_time_features(combined_df, time_numeric)
            row = combined_df.iloc[-1]
        else:
            row = current_df.assign(time_since_last=0).iloc[0]
        current_df = pd.get_dummies(current_df, columns=categorical_features)
        current_df = current_df.reindex(columns=lat_expected_features, fill_value=0)
        X_new_lat = pd.DataFrame(current_df[lat_expected_features].fillna(0), columns=lat_expected_features)
        X_new_lat_scaled = scaler_latitude.transform(X_new_lat)
        pred_lat = latitude_model.predict(X_new_lat_scaled, verbose=0)
        return pred_lat[0][0], True
    except Exception as e:
        print(f"Latitude prediction failed for ID {earthquake_id}: {e}")
        return np.nan, False

def process_earthquake_data_for_longitude(current_df, earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df):
    try:
        current_df['year'] = event_time.year
        current_df['month'] = event_time.month
        current_df['day'] = event_time.day
        current_df['lat_lon_interaction'] = current_df['latitude'] * current_df['longitude']
        if not historical_df.empty:
            combined_df = pd.concat([historical_df, current_df]).reset_index(drop=True)
            combined_df = compute_time_features(combined_df, time_numeric)
            row = combined_df.iloc[-1]
        else:
            row = current_df.assign(time_since_last=0).iloc[0]
        current_df = pd.get_dummies(current_df, columns=categorical_features)
        current_df = current_df.reindex(columns=lon_expected_features, fill_value=0)
        X_new_lon = pd.DataFrame(current_df[lon_expected_features].fillna(0), columns=lon_expected_features)
        X_new_lon_scaled = scaler_longitude.transform(X_new_lon)
        pred_lon = longitude_model.predict(X_new_lon_scaled, verbose=0)
        return pred_lon[0][0], True
    except Exception as e:
        print(f"Longitude prediction failed for ID {earthquake_id}: {e}")
        return np.nan, False

def process_earthquake_data_for_depth(current_df, earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df):
    try:
        # Add depth as a column to current_df for consistency
        current_df = current_df.assign(depth=depth)

        # Add depth indicators based on the scalar depth value
        current_df['shallow_depth_indicator'] = int(depth < 70)
        current_df['mid_depth_indicator'] = int((depth >= 100) and (depth <= 300))
        current_df['deep_depth_indicator'] = int(depth > 300)

        current_df['year'] = event_time.year
        current_df['month'] = event_time.month
        current_df['day'] = event_time.day
        current_df['lat_lon_interaction'] = current_df['latitude'] * current_df['longitude']

        if not historical_df.empty:
            combined_df = pd.concat([historical_df, current_df]).reset_index(drop=True)
            combined_df = compute_time_features(combined_df, time_numeric)
            row = combined_df.iloc[-1]
        else:
            row = current_df.assign(time_since_last=0).iloc[0]

        # Apply one-hot encoding for categorical features
        current_df = pd.get_dummies(current_df, columns=categorical_features)

        # Reindex to match expected features, preserving existing values
        current_df = current_df.reindex(columns=depth_expected_features, fill_value=0)

        # Prepare and scale the input data
        X_new_depth = pd.DataFrame(current_df[depth_expected_features].fillna(0), columns=depth_expected_features)
        X_new_depth_scaled = scaler_depth.transform(X_new_depth)

        # Predict and process depth
        pred_depth_log = depth_model.predict(X_new_depth_scaled, verbose=0)[0][0]
        pred_depth = np.expm1(pred_depth_log)  # Convert from log scale
        pred_depth = np.clip(pred_depth, 0, 700)  # Clip to realistic range

        # Apply post-processing corrections
        mid_threshold_low = 100
        mid_threshold_high = 300
        deep_threshold = 300
        shallow_mask = depth < mid_threshold_low
        mid_mask = (depth >= mid_threshold_low) and (depth <= mid_threshold_high)
        deep_mask = depth > deep_threshold
        correction_factor_shallow = 0.7
        correction_factor_mid = 0.6
        correction_factor_deep = 0.5
        errors = depth - pred_depth
        pred_depth = pred_depth + (correction_factor_shallow * errors if shallow_mask else
                                 correction_factor_mid * errors if mid_mask else
                                 correction_factor_deep * errors if deep_mask else 0)

        return pred_depth, True
    except Exception as e:
        print(f"Depth prediction failed for ID {earthquake_id}: {e}")
        return np.nan, False
    
def process_earthquake_data_for_magnitude(current_df, earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df):
    try:
        current_df['year'] = event_time.year
        current_df['month'] = event_time.month
        current_df['day'] = event_time.day
        current_df['rms'] = 0
        current_df['horizontalError'] = 0
        current_df['depthError'] = 0
        current_df['magError'] = 0
        current_df['magNst'] = 0
        current_df['dist_to_fault'] = 0
        current_df['time_numeric'] = time_numeric
        if not historical_df.empty:
            combined_df = pd.concat([historical_df, current_df]).reset_index(drop=True)
            combined_df = compute_time_features(combined_df, time_numeric)
            row = combined_df.iloc[-1]
        else:
            row = current_df.assign(time_since_last=0).iloc[0]
        current_df = pd.get_dummies(current_df, columns=categorical_features)
        current_df = current_df.reindex(columns=mag_expected_features, fill_value=0)
        X_new_mag = pd.DataFrame(current_df[mag_expected_features].fillna(0), columns=mag_expected_features)
        X_new_mag_scaled = scaler_magnitude.transform(X_new_mag)
        pred_mag = magnitude_model.predict(X_new_mag_scaled, verbose=0)
        return pred_mag[0][0], True
    except Exception as e:
        print(f"Magnitude prediction failed for ID {earthquake_id}: {e}")
        return np.nan, False

def create_local_sequences(X, y, coords, sequence_length=10, radius_km=200):
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

@tf.function(reduce_retracing=True)
def predict_time_difference(model, x):
    return model(x, training=False)

def process_earthquake_data_for_time(current_df, earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df):
    try:
        sequence_length = 10
        current_df['year'] = event_time.year
        current_df['month'] = event_time.month
        current_df['day'] = event_time.day
        current_df['lat_lon_interaction'] = current_df['latitude'] * current_df['longitude']
        
        all_categorical_values = {
            'magType': ['m', 'mb', 'mb_lg', 'md', 'ml', 'ms', 'mw', 'mwb', 'mwc', 'mwr', 'mww'],
            'net': ['iscgem', 'official', 'us'],
            'type': ['earthquake'],
            'status': ['reviewed'],
            'locationSource': ['guc', 'iscgem', 'lim', 'sja', 'unm', 'us', 'us_guc', 'us_sja'],
            'magSource': ['ath', 'gcmt', 'guc', 'hrv', 'iscgem', 'official', 'sja', 'unm', 'us', 'us_guc', 'us_sja']
        }
        
        for col, values in all_categorical_values.items():
            for val in values:
                current_df[f'{col}_{val}'] = (current_df[col] == val).astype(int)
        
        if not historical_df.empty:
            combined_df = pd.concat([historical_df, current_df]).reset_index(drop=True)
            combined_df = compute_time_features(combined_df, time_numeric)
            if len(combined_df) < sequence_length:
                padding_df = pd.DataFrame(0, index=range(sequence_length - len(combined_df)),
                                        columns=combined_df.columns)
                combined_df = pd.concat([padding_df, combined_df]).reset_index(drop=True)
            X = combined_df[time_expected_features].values
            y = combined_df['time_to_next'].values if 'time_to_next' in combined_df else np.zeros(len(combined_df))
            coords = combined_df[['latitude', 'longitude']].values
            X_seq, _, _ = create_local_sequences(X, y, coords, sequence_length)
            if len(X_seq) == 0:
                raise ValueError("No valid sequences created")
            # Flatten the sequence and convert to DataFrame with feature names
            X_seq_2d = X_seq.reshape(-1, X_seq.shape[-1])
            X_seq_df = pd.DataFrame(X_seq_2d, columns=time_expected_features)
            X_seq_scaled = scaler_time.transform(X_seq_df)
            X_seq_scaled = X_seq_scaled.reshape(1, sequence_length, -1)
        else:
            combined_df = current_df.copy()
            combined_df = combined_df.reindex(columns=time_expected_features, fill_value=0)
            X = np.zeros((sequence_length, len(time_expected_features)))
            X[-1] = combined_df[time_expected_features].values
            X_seq = X.reshape(1, sequence_length, -1)
            # Flatten and convert to DataFrame with feature names
            X_seq_2d = X_seq.reshape(-1, X_seq.shape[-1])
            X_seq_df = pd.DataFrame(X_seq_2d, columns=time_expected_features)
            X_seq_scaled = scaler_time.transform(X_seq_df)
            X_seq_scaled = X_seq_scaled.reshape(1, sequence_length, -1)

        pred_time_diff = predict_time_difference(time_model, tf.convert_to_tensor(X_seq_scaled, dtype=tf.float32))[0][0]
        
        # Use event_time as the base time instead of current_time
        pred_time_diff = max(0, min(365, float(pred_time_diff)))  # Cap at 365 days
        total_seconds = pred_time_diff * 24 * 3600
        predicted_time = event_time + timedelta(seconds=total_seconds)
        return predicted_time, True
    except Exception as e:
        print(f"Time prediction failed for ID {earthquake_id}: {e}")
        return pd.NaT, False

def check_prediction_matches(predictions_df, new_event):
    earthquake_id = new_event['id']
    properties = new_event['properties']
    geometry = new_event['geometry']
    lat = geometry['coordinates'][1]
    lon = geometry['coordinates'][0]
    mag = properties['mag'] if 'mag' in properties else 0
    depth = geometry['coordinates'][2] if len(geometry['coordinates']) > 2 else 0
    time_ms = properties['time']
    event_time = pd.to_datetime(time_ms, unit='ms').tz_localize(None)

    LAT_THRESHOLD = 0.5
    LON_THRESHOLD = 0.5
    DEPTH_THRESHOLD = 50
    MAG_THRESHOLD = 0.5
    TIME_THRESHOLD = timedelta(hours=24)

    unmatched_predictions = predictions_df[predictions_df['prediction_correct'].isna()]
    if unmatched_predictions.empty:
        return predictions_df

    for idx, pred in unmatched_predictions.iterrows():
        pred_time = pd.to_datetime(pred['prediction_timestamp'])
        if not (pred_time < event_time <= pred_time + TIME_THRESHOLD):
            continue
        if (abs(pred['predicted_latitude'] - lat) <= LAT_THRESHOLD and
            abs(pred['predicted_longitude'] - lon) <= LON_THRESHOLD and
            abs(pred['predicted_depth'] - depth) <= DEPTH_THRESHOLD and
            abs(pred['predicted_magnitude'] - mag) <= MAG_THRESHOLD):
            predictions_df.at[idx, 'prediction_correct'] = True
            predictions_df.at[idx, 'predicted_earthquake_id'] = earthquake_id
            print(f"Prediction for ID {pred['earthquake_id']} matched with Earthquake ID {earthquake_id}")
            break

    current_time = datetime.now()
    for idx, pred in predictions_df.iterrows():
        if pd.isna(pred['prediction_correct']):
            pred_time = pd.to_datetime(pred['prediction_timestamp'])
            if current_time - pred_time > TIME_THRESHOLD:
                predictions_df.at[idx, 'prediction_correct'] = False
                predictions_df.at[idx, 'predicted_earthquake_id'] = np.nan
                print(f"Prediction for ID {pred['earthquake_id']} expired without a match.")

    return predictions_df

def process_earthquake_data(earthquake_data, historical_df):
    global predictions_df, predicted_earthquakes
    new_data = []
    reference_date = pd.Timestamp('1970-01-01')

    try:
        predictions_df = pd.read_csv(PREDICTIONS_CSV_FILE)
        print(f"Loaded {len(predictions_df)} existing predictions from {PREDICTIONS_CSV_FILE}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        predictions_df = pd.DataFrame(columns=['earthquake_id', 'latitude', 'longitude', 'depth', 'predicted_latitude',
                                              'predicted_longitude', 'predicted_depth', 'predicted_magnitude',
                                              'predicted_time', 'prediction_timestamp', 'predicted_earthquake_id',
                                              'prediction_correct'])
        predictions_df.to_csv(PREDICTIONS_CSV_FILE, index=False)

    for feature in earthquake_data:
        predictions_df = check_prediction_matches(predictions_df, feature)
    predictions_df.to_csv(PREDICTIONS_CSV_FILE, index=False, date_format='%Y-%m-%d')

    print(f"Processing {len(earthquake_data)} earthquake features")
    for feature in earthquake_data:
        earthquake_id = feature['id']
        if earthquake_id in predictions_df['earthquake_id'].values:
            print(f"Skipping duplicate ID: {earthquake_id}")
            continue

        properties = feature['properties']
        geometry = feature['geometry']
        lat = geometry['coordinates'][1]
        lon = geometry['coordinates'][0]
        mag = properties['mag'] if 'mag' in properties else 0
        depth = geometry['coordinates'][2] if len(geometry['coordinates']) > 2 else 0
        time_ms = properties['time']
        event_time = pd.to_datetime(time_ms, unit='ms').tz_localize(None)
        time_numeric = (event_time - reference_date).total_seconds() / (24 * 3600)

        if not (MIN_LAT <= lat <= MAX_LAT and MIN_LON <= lon <= MAX_LON):
            print(f"Skipping out-of-region earthquake ID: {earthquake_id} at {lat}, {lon}")
            continue

        current_df = pd.DataFrame({
            'id': [earthquake_id],
            'latitude': [lat],
            'longitude': [lon],
            'depth': [depth],
            'time_numeric': [time_numeric],
            'magnitude': [mag],
            'magType': [properties.get('magType', 'unknown')],
            'net': [properties.get('net', 'unknown')],
            'type': [properties.get('type', 'unknown')],
            'status': [properties.get('status', 'unknown')],
            'locationSource': [properties.get('locationSource', 'unknown')],
            'magSource': [properties.get('magSource', 'unknown')],
            'nst': [properties.get('nst', np.nan)],
            'gap': [properties.get('gap', np.nan)],
            'dmin': [properties.get('dmin', np.nan)],
            'rms': [properties.get('rms', 0)],
            'horizontalError': [properties.get('horizontalError', 0)],
            'depthError': [properties.get('depthError', 0)],
            'magError': [properties.get('magError', 0)],
            'magNst': [properties.get('magNst', 0)]
        })

        pred_lat, lat_success = process_earthquake_data_for_latitude(current_df.copy(), earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df)
        pred_lon, lon_success = process_earthquake_data_for_longitude(current_df.copy(), earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df)
        pred_depth, depth_success = process_earthquake_data_for_depth(current_df.copy(), earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df)
        pred_mag, mag_success = process_earthquake_data_for_magnitude(current_df.copy(), earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df)
        pred_time, time_success = process_earthquake_data_for_time(current_df.copy(), earthquake_id, lat, lon, depth, time_numeric, event_time, historical_df)

        if not (lat_success and lon_success and depth_success and mag_success and time_success):
            print(f"Skipping prediction for ID {earthquake_id} due to failed prediction(s). Details: lat={lat_success}, lon={lon_success}, depth={depth_success}, mag={mag_success}, time={time_success}")
            continue

        raw_data = {
            'id': earthquake_id,  # Add ID to raw data
            'time': event_time.isoformat(),
            'latitude': lat,
            'longitude': lon,
            'depth': depth,
            'magnitude': mag
        }
        new_data.append(raw_data)

        new_prediction = pd.DataFrame({
            'earthquake_id': [earthquake_id],
            'latitude': [lat],
            'longitude': [lon],
            'depth': [depth],
            'predicted_latitude': [pred_lat],
            'predicted_longitude': [pred_lon],
            'predicted_depth': [pred_depth],
            'predicted_magnitude': [pred_mag],
            'predicted_time': [pred_time.strftime('%Y-%m-%d %H:%M:%S')],  # Updated format
            'prediction_timestamp': [datetime.now()],
            'predicted_earthquake_id': [np.nan],
            'prediction_correct': [np.nan]
        })
        predictions_df = pd.concat([predictions_df, new_prediction]).reset_index(drop=True)
        predictions_df.to_csv(PREDICTIONS_CSV_FILE, index=False, date_format='%Y-%m-%d')

        # Store prediction for web map
        global predicted_earthquakes
        prediction = {
            "id": earthquake_id,
            "latitude": pred_lat,
            "longitude": pred_lon,
            "depth": pred_depth,
            "magnitude": pred_mag,
            "time": pred_time.strftime('%Y-%m-%d %H:%M:%S')  # Updated format
        }
        predicted_earthquakes.append(prediction)
        print(f"Added prediction for ID {earthquake_id} to predicted_earthquakes. Total: {len(predicted_earthquakes)}")

        print(f"Earthquake ID: {earthquake_id} at {lat}, {lon}, Depth {depth}:")
        print(f"  Predicted Latitude: {pred_lat:.4f}")
        print(f"  Predicted Longitude: {pred_lon:.4f}")
        print(f"  Predicted Depth: {pred_depth:.4f}")
        print(f"  Predicted Magnitude: {pred_mag:.4f}")
        print(f"  Predicted Time: {pred_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if new_data:
        df_new = pd.DataFrame(new_data)
        if os.path.exists(API_CSV_FILE):
            existing_df = pd.read_csv(API_CSV_FILE)
            api_df = pd.concat([existing_df, df_new]).drop_duplicates(subset=['id', 'time', 'latitude', 'longitude']).reset_index(drop=True)
        else:
            api_df = df_new
        api_df.to_csv(API_CSV_FILE, index=False)
        print(f"Updated {len(api_df)} earthquakes in {API_CSV_FILE}")
        return df_new
    return pd.DataFrame()

def display_timer():
    for remaining in range(120, 0, -1):
        mins, secs = divmod(remaining, 60)
        print(f"Waiting {mins:02d}:{secs:02d} until next fetch...", end='\r')
        time.sleep(1)
    print("Fetching new data...              ")

@app.route('/')
def index():
    print(f"Rendering map with {len(predicted_earthquakes)} predicted and {len(api_earthquakes)} actual earthquakes")
    return render_template('index.html', predicted_earthquakes=predicted_earthquakes, api_earthquakes=api_earthquakes)

@app.route('/predictions')
def get_predictions():
    return jsonify(predicted_earthquakes)

def start_web_server():
    fetch_and_save_api_data()  # Initial fetch and save
    app.run(debug=True, host='0.0.0.0', port=5000)

historical_df = pd.DataFrame(columns=['latitude', 'longitude', 'magnitude', 'depth', 'time_numeric'])
while True:
    print(f"\nFetching data at {datetime.now()}")
    earthquake_data = fetch_earthquake_data()
    if earthquake_data:
        new_df = process_earthquake_data(earthquake_data, historical_df)
        if not new_df.empty:
            historical_df = pd.concat([historical_df, new_df]).drop_duplicates().reset_index(drop=True)
    else:
        print("No new earthquake data available.")
    display_timer()