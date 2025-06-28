from flask import Flask, render_template, jsonify
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Load API earthquake data
api_csv_path = 'api_earthquakes.csv'
api_earthquakes = []
if os.path.exists(api_csv_path):
    api_df = pd.read_csv(api_csv_path)
    api_earthquakes = api_df.to_dict('records')
    for quake in api_earthquakes:
        quake['coordinates'] = f"{quake['latitude']}, {quake['longitude']}"
        # Ensure time is a string for consistency
        quake['time'] = str(quake['time'])

# Load predicted earthquake data from CSV
predictions_csv_path = 'earthquake_predictions.csv'
predicted_earthquakes = []
if os.path.exists(predictions_csv_path):
    predictions_df = pd.read_csv(predictions_csv_path)
    predicted_earthquakes = predictions_df.to_dict('records')
    for p_quake in predicted_earthquakes:
        p_quake['coordinates'] = f"{p_quake['predicted_latitude']}, {p_quake['predicted_longitude']}"
        # Ensure time is a string for consistency
        p_quake['time'] = str(p_quake['predicted_time'])
        # if pd.notna(row['predicted_latitude']):  # Only include rows with valid predictions
        #     predicted_earthquakes.append({
        #         "id": row['earthquake_id'],
        #         "coordinates": f"{row['predicted_latitude']}, {row['predicted_longitude']}",
        #         "depth": row['predicted_depth'],
        #         "magnitude": row['predicted_magnitude'],
        #         "time": row['predicted_time']
        #     })

@app.route('/')
def index():
    return render_template('index.html', predicted_earthquakes=predicted_earthquakes, api_earthquakes=api_earthquakes)

@app.route('/predictions')
def get_predictions():
    return jsonify(predicted_earthquakes)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)