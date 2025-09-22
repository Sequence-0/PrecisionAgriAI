import os
import numpy as np
import lightgbm as lgb
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import skimage.io as skio
import torch
import subprocess
import json
import matplotlib
matplotlib.use('Agg') # Set Matplotlib backend to Agg
import matplotlib.pyplot as plt
import time
from threading import Lock
import warnings # Added for normHSI function

# --- Global Lock ---
calculation_lock = Lock()

# Define the normHSI function
def normHSI(R, eps=1e-6, stat=False):
    if isinstance(R, torch.Tensor):
        rmax, rmin = torch.max(R), torch.min(R)
        R = (R - rmin)/(rmax - rmin + eps)
    elif isinstance(R, np.ndarray):
        rmax, rmin = np.max(R), np.min(R)
        R = (R - rmin)/(rmax - rmin + eps)
    else:
        warnings.warn("Unsupport data type of input HSI")
        return
    if stat:
        return R, rmax, rmin
    return R

# --- Index Calculation Functions ---
def calculate_ndvi(image_data, timestamp):
    """Calculates NDVI and returns a colormapped image path."""
    with calculation_lock:
        # Bands determined from notebook: Red ~670nm (Band 45), NIR ~800nm (Band 78)
        red_band = image_data[45, :, :]
        nir_band = image_data[78, :, :]
        
        # Calculate NDVI
        denom = (nir_band + red_band)
        denom[denom == 0] = 1e-6
        ndvi = (nir_band - red_band) / denom
        ndvi = np.clip(ndvi, -1.0, 1.0)
        
        # Save NDVI image
        plt.figure()
        plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(label='NDVI')
        plt.title('NDVI')
        ndvi_filename = f'ndvi_{timestamp}.png'
        ndvi_path = os.path.join('static', 'generated', ndvi_filename)
        plt.savefig(ndvi_path)
        plt.close()
        # Calculate mean NDVI
        ndvi_mean = np.mean(ndvi)
        return ndvi_path, ndvi_mean

def calculate_rx_anomaly(image_data, timestamp):
    """Calculates RX Anomaly and returns a colormapped image path."""
    with calculation_lock:
        H, W, C = image_data.shape
        X = image_data.reshape(-1, C).astype(np.float32)

        # Standardize for stability
        X_mean = X.mean(axis=0, keepdims=True)
        X_centered = X - X_mean
        # Covariance + regularization
        cov = np.cov(X_centered, rowvar=False)
        # Add small ridge to avoid singular covariance
        cov += np.eye(cov.shape[0], dtype=np.float32) * 1e-6
        inv_cov = np.linalg.pinv(cov)  # robust pseudo-inverse

        # RX score: (x-μ)^T Σ^{-1} (x-μ)
        r = np.einsum('ij,jk,ik->i', X_centered, inv_cov, X_centered)
        r = r.reshape(H, W)

        # Normalize for visualization
        r = (r - r.min()) / (r.max() - r.min() + 1e-8)
        
        # Save RX Anomaly image
        plt.figure()
        plt.imshow(r, cmap='hot', vmin=0, vmax=1)
        plt.colorbar(label='RX anomaly score (0-1)')
        plt.title('RX Anomaly Detection')
        rx_filename = f'rx_anomaly_{timestamp}.png'
        rx_path = os.path.join('static', 'generated', rx_filename)
        plt.savefig(rx_path)
        plt.close()
        return rx_path

def generate_ndvi_insights(ndvi_mean):
    if ndvi_mean > 0.6:
        return "High NDVI indicates healthy and vigorous crop growth. This suggests good plant health and photosynthetic activity."
    elif ndvi_mean > 0.4:
        return "Moderate NDVI suggests average crop health. There might be some areas of stress or less vigorous growth."
    else:
        return "Low NDVI indicates stressed or unhealthy crop. This could be due to disease, nutrient deficiency, or water stress."

def generate_disease_insights(prediction_result):
    if prediction_result == 'Serious-FHB':
        return "Serious FHB disease detected. Immediate action is recommended to prevent further spread and significant yield loss. Consider fungicide application and removal of infected plants."
    elif prediction_result == 'Mild-FHB':
        return "Mild FHB disease detected. Monitor the crop closely and consider preventative measures. Early intervention can limit disease progression."
    else:
        return "No significant FHB disease detected. Continue regular monitoring and maintain good agricultural practices."

# --- Model Training and Feature Extraction ---
def get_top_features_and_train_model():
    """
    Runs the training script, captures the top 30 feature indices,
    and ensures the model is trained and saved.
    """
    try:
        # Execute the training script
        result = subprocess.run(
            ['python', 'train_model.py'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract the top 30 features from the script's output
        output_lines = result.stdout.strip().split('\n')
        top_features_line = [line for line in output_lines if "Top 30 Feature Indices:" in line]
        
        if not top_features_line:
            raise RuntimeError("Could not find top feature indices in train_model.py output.")
            
        # The line is expected to be "Top 30 Feature Indices: [..., ..., ...]"
        # We need to parse the list from this string
        features_str = top_features_line[0].split(':', 1)[1].strip()
        top_30_features = json.loads(features_str)
        
        print(f"Successfully extracted top 30 features: {top_30_features}")
        return top_30_features

    except subprocess.CalledProcessError as e:
        print(f"Error running train_model.py: {e}")
        print(f"Stderr: {e.stderr}")
        # Fallback to placeholder if training fails
        return [i for i in range(30)]
    except (RuntimeError, IndexError, json.JSONDecodeError) as e:
        print(f"Error parsing training script output: {e}")
        # Fallback to placeholder if parsing fails
        return [i for i in range(30)]

# Get features and train the model on startup
top_30_features = get_top_features_and_train_model()
# Load the trained model
model = lgb.Booster(model_file='model.txt')

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_fallback_secret_key_for_dev') # Use environment variable for production

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insights')
def insights():
    last_data = session.get('last_prediction_data')
    return render_template('insights.html', last_data=last_data)

@app.route('/get_last_prediction_data', methods=['GET'])
def get_last_prediction_data():
    last_data = session.get('last_prediction_data')
    if last_data:
        return jsonify(last_data)
    return jsonify({'error': 'No previous prediction data found'}), 404

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)

        # Preprocess the image
        img = skio.imread(filepath)
        img_float = img.astype('float')[:,:,:125]
        img_tensor = torch.tensor(img_float).permute(2,0,1)
        img_norm = normHSI(img_tensor)
        
        # --- Calculate Indices ---
        timestamp = int(time.time())
        # Permute for RX anomaly calculation
        img_for_rx = img_norm.permute(1, 2, 0).numpy()
        ndvi_path, ndvi_mean = calculate_ndvi(img_norm.numpy(), timestamp)
        rx_path = calculate_rx_anomaly(img_for_rx, timestamp)
        
        # --- Prediction ---
        spec = []
        for i in range(img_norm.shape[0]):
            mean = torch.mean(img_norm[i,:,:])
            spec.append(mean.item())
        
        selected_spec = [spec[i] for i in top_30_features]
        
        prediction = model.predict([selected_spec])
        
        result = 'Mild-FHB' if prediction[0] > 0.5 else 'Serious-FHB'
        
        # Generate insights
        ndvi_insights = generate_ndvi_insights(ndvi_mean)
        disease_insights = generate_disease_insights(result)

        # Store data in session
        session['last_prediction_data'] = {
            'original_filename': filename,
            'original_filepath': filepath, # Store the path to the original uploaded file
            'prediction': result,
            'ndvi_mean': ndvi_mean,
            'ndvi_image': ndvi_path,
            'rx_anomaly_image': rx_path,
            'ndvi_insights': ndvi_insights,
            'disease_insights': disease_insights
        }
        
        return jsonify({
            'prediction': result,
            'ndvi_image': ndvi_path,
            'rx_anomaly_image': rx_path,
            'ndvi_mean': ndvi_mean,
            'ndvi_insights': ndvi_insights,
            'disease_insights': disease_insights
        })

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    os.makedirs(os.path.join('static', 'generated'), exist_ok=True)
    app.run(debug=True)
